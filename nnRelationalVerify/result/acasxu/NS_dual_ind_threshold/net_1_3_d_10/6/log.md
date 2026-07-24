## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 19342.684212278527


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125)
1: (-12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062)
2: (-18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625)
3: (-6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750)
4: (-20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 2.33 = 4.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -19343.0710737, upper bound: 19343.0710737

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8719469, upper bound: 19342.9859180
time: 0.91 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8694456, upper bound: 19342.8694456
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 3, lower bound: -19342.8719469, upper bound: 19342.9859180
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 3, lower bound: -19342.8694456, upper bound: 19342.8694456

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -15606.2392578, 11546.4335938, -15747.2714844, 11651.2080078, -27257.4472656, 27293.7050781
1: -12631.3779297, 11177.4941406, -12745.3994141, 11278.1923828, -23909.5683594, 23922.8925781
2: -18419.4785156, 12209.6884766, -18586.4863281, 12319.7011719, -30739.1777344, 30796.1757812
3: -6374.2680664, 17344.5468750, -6431.4116211, 17502.5371094, -23876.8046875, 23775.9570312
4: -20242.6933594, 12071.0019531, -20426.1640625, 12180.4560547, -32423.1445312, 32497.1660156

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4426269, upper bound: 19342.5671446
time: 0.93 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8593109, upper bound: 19342.9413474
time: 0.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -17825.7265625, 13106.9628906, -15596.7001953, 11535.9609375, -29361.6875000, 28703.6640625
1: -14429.7958984, 12691.0000000, -12623.6435547, 11165.6630859, -25595.4589844, 25314.6445312
2: -21053.4453125, 13860.3105469, -18410.4707031, 12196.4794922, -33249.9257812, 32270.7812500
3: -7247.8081055, 19801.2207031, -6367.0297852, 17334.3710938, -24582.1796875, 26168.2500000
4: -23130.5800781, 13698.3564453, -20232.4316406, 12058.6171875, -35189.1953125, 33930.7851562

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4420703, upper bound: 19342.5386763
time: 0.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8527299, upper bound: 19342.8527299
time: 1.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.70 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.70
Output dim: 3, lower bound: -19342.4426269, upper bound: 19342.5671446
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.70
Output dim: 3, lower bound: -19342.8593109, upper bound: 19342.9413474
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 3.70
Output dim: 3, lower bound: -19342.4420703, upper bound: 19342.5386763
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.70
Output dim: 3, lower bound: -19342.8527299, upper bound: 19342.8527299

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -15316.4482422, 11336.1972656, -16259.4873047, 12067.2714844, -27383.7187500, 27595.6835938
1: -12396.1357422, 10975.1064453, -13161.1318359, 11684.9101562, -24081.0468750, 24136.2382812
2: -18080.8984375, 11989.7812500, -19197.7636719, 12762.2197266, -30843.1152344, 31187.5429688
3: -6258.8823242, 17027.0703125, -6659.0791016, 18118.0644531, -24376.9472656, 23686.1484375
4: -19869.4589844, 11852.2119141, -21090.9082031, 12606.7187500, -32476.1757812, 32943.1171875

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7815316, upper bound: 19342.9282865
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8589996, upper bound: 19342.9412529
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8593109, upper bound: 19342.9412529
time: 1.02 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -17551.5761719, 12908.9990234, -16070.5917969, 11926.1865234, -29477.7558594, 28979.5878906
1: -14207.7412109, 12500.7402344, -13008.7939453, 11547.0849609, -25754.8222656, 25509.5312500
2: -20733.3554688, 13653.2080078, -18977.4218750, 12611.0410156, -33344.3945312, 32630.6289062
3: -7137.7939453, 19500.3984375, -6579.2919922, 17908.1425781, -25045.9355469, 26079.6875000
4: -22777.0625000, 13492.4091797, -20847.9570312, 12457.1328125, -35234.1914062, 34340.3671875

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7790830, upper bound: 19342.8474634
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7748636, upper bound: 19342.7748636
time: 0.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.72 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 3, lower bound: -19342.8589996, upper bound: 19342.9412529
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 3, lower bound: -19342.8593109, upper bound: 19342.9412529
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 3, lower bound: -19342.7790830, upper bound: 19342.8474634
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 3, lower bound: -19342.7748636, upper bound: 19342.7748636

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14646.3906250, 10858.1074219, -15983.6728516, 11868.8359375, -26515.2246094, 26841.7812500
1: -11853.3164062, 10510.5703125, -12937.8867188, 11492.0937500, -23345.4101562, 23448.4531250
2: -17290.3300781, 11483.5751953, -18872.5585938, 12552.6621094, -29842.9921875, 30356.1328125
3: -5991.3754883, 16284.8320312, -6548.1894531, 17812.3808594, -23803.7558594, 22833.0214844
4: -18999.7871094, 11351.3232422, -20732.1933594, 12398.9892578, -31398.7773438, 32083.5156250

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8469451, upper bound: 19342.9342982
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8302806, upper bound: 19342.9271370
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -15217.4482422, 11261.2529297, -16195.8271484, 12020.1015625, -27237.5468750, 27457.0800781
1: -12316.3398438, 10902.6767578, -13109.7373047, 11639.0302734, -23955.3710938, 24012.4140625
2: -17964.7187500, 11910.8496094, -19123.0937500, 12711.9628906, -30676.6816406, 31033.9433594
3: -6217.8437500, 16916.9863281, -6632.4633789, 18047.5703125, -24265.4140625, 23549.4492188
4: -19741.6308594, 11773.4121094, -21008.7636719, 12557.1591797, -32298.7890625, 32782.1757812

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8469502, upper bound: 19342.9293872
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8292801, upper bound: 19342.9232026
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17286.1289062, 12708.2949219, -15869.3076172, 11779.1972656, -29065.3261719, 28577.6015625
1: -13992.7949219, 12307.3007812, -12845.7236328, 11405.0859375, -25397.8808594, 25153.0234375
2: -20419.6757812, 13439.9873047, -18739.5800781, 12455.4316406, -32875.1054688, 32179.5625000
3: -7029.5825195, 19204.4375000, -6497.3706055, 17684.7500000, -24714.3300781, 25701.8085938
4: -22432.7285156, 13280.5439453, -20586.8652344, 12302.9257812, -34735.6562500, 33867.4062500

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7772943, upper bound: 19342.8472954
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7790830, upper bound: 19342.8474634
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -18846.7753906, 13812.0195312, -15646.1425781, 11603.5722656, -30450.3476562, 29458.1621094
1: -15254.1708984, 13380.9794922, -12666.8320312, 11236.2216797, -26490.3847656, 26047.8125000
2: -22240.4960938, 14610.1201172, -18475.9746094, 12271.0957031, -34511.5937500, 33086.0937500
3: -7631.0434570, 20919.4023438, -6400.5605469, 17429.3828125, -25060.4257812, 27319.9628906
4: -24424.4921875, 14430.3857422, -20293.8144531, 12119.5410156, -36544.0273438, 34724.1953125

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7730825, upper bound: 19342.7746953
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7748636, upper bound: 19342.7748636
time: 1.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.95 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8469451, upper bound: 19342.9342982
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8302806, upper bound: 19342.9271370
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8469502, upper bound: 19342.9293872
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8292801, upper bound: 19342.9232026
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.7772943, upper bound: 19342.8472954
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.7790830, upper bound: 19342.8474634
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.7730825, upper bound: 19342.7746953
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.7748636, upper bound: 19342.7748636

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14540.6298828, 10778.8388672, -15756.9433594, 11699.3710938, -26239.9960938, 26535.7792969
1: -11767.7939453, 10433.7294922, -12754.5234375, 11328.3632812, -23096.1542969, 23188.2500000
2: -17165.2207031, 11399.6406250, -18604.3281250, 12373.6484375, -29538.8691406, 30003.9687500
3: -5947.7441406, 16166.1503906, -6454.3774414, 17558.2460938, -23505.9902344, 22620.5273438
4: -18862.4296875, 11268.5693359, -20437.7871094, 12222.4384766, -31084.8652344, 31706.3554688

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8190363, upper bound: 19342.8674572
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8274170, upper bound: 19342.9172722
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14486.6767578, 10738.2900391, -17838.9003906, 13232.8984375, -27719.5722656, 28577.1855469
1: -11724.1289062, 10394.7636719, -14442.9609375, 12814.6308594, -24538.7578125, 24837.7246094
2: -17102.0859375, 11356.9492188, -21065.7148438, 13996.4716797, -31098.5585938, 32422.6621094
3: -5924.5297852, 16106.3408203, -7303.9252930, 19891.6093750, -25816.1386719, 23410.2656250
4: -18792.8828125, 11226.0546875, -23144.0136719, 13835.2636719, -32628.1425781, 34370.0703125

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8036989, upper bound: 19342.8623491
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8133079, upper bound: 19342.9101610
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15112.1523438, 11182.1298828, -15969.5117188, 11850.9404297, -26963.0937500, 27151.6406250
1: -12231.2275391, 10826.2431641, -12926.7529297, 11475.5478516, -23706.7714844, 23752.9941406
2: -17840.2363281, 11827.5039062, -18855.4511719, 12533.2275391, -30373.4628906, 30682.9550781
3: -6174.2807617, 16798.8906250, -6538.7832031, 17794.0292969, -23968.3105469, 23337.6718750
4: -19604.9589844, 11690.8535156, -20714.9453125, 12380.8281250, -31985.7871094, 32405.7949219

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8193230, upper bound: 19342.8646660
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8274282, upper bound: 19342.9144966
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15073.7656250, 11153.8730469, -18030.1953125, 13374.7607422, -28448.5273438, 29184.0683594
1: -12200.1250000, 10798.9335938, -14597.7050781, 12952.2324219, -25152.3554688, 25396.6308594
2: -17795.4335938, 11797.8261719, -21292.3691406, 14146.2451172, -31941.6796875, 33090.1953125
3: -6158.3950195, 16756.7128906, -7383.5170898, 20106.2968750, -26264.6855469, 24140.2304688
4: -19555.6035156, 11661.2041016, -23394.4589844, 13983.5253906, -33539.1289062, 35055.6601562

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8030434, upper bound: 19342.8596249
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8122956, upper bound: 19342.9079102
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17029.0410156, 12528.1611328, -15221.5097656, 11314.7265625, -28343.7675781, 27749.6718750
1: -13784.4111328, 12131.8955078, -12321.5878906, 10952.9765625, -24737.3847656, 24453.4843750
2: -20116.4765625, 13249.3779297, -17974.9902344, 11963.1494141, -32079.6250000, 31224.3652344
3: -6926.8686523, 18921.0761719, -6238.8300781, 16966.2089844, -23893.0742188, 25159.9023438
4: -22098.4062500, 13091.2158203, -19745.3769531, 11815.9931641, -33914.3984375, 32836.5937500

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8344086
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8184497
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17207.2500000, 12648.2607422, -15775.4414062, 11709.4628906, -28916.7128906, 28423.7031250
1: -13929.1494141, 12248.6875000, -12769.9326172, 11337.5175781, -25266.6640625, 25018.6210938
2: -20326.9199219, 13376.3535156, -18629.4589844, 12381.1523438, -32708.0683594, 32005.8125000
3: -6996.7275391, 19115.8496094, -6458.0712891, 17580.7343750, -24577.4609375, 25573.9218750
4: -22330.7246094, 13217.6425781, -20465.7460938, 12229.6699219, -34560.3867188, 33683.3789062

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8345835
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8182567
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -18621.6074219, 13649.9843750, -14991.4648438, 11134.3759766, -29755.9804688, 28641.4492188
1: -15072.4082031, 13222.6259766, -12137.1142578, 10779.9023438, -25852.3105469, 25359.7402344
2: -21975.1972656, 14437.8320312, -17703.4003906, 11774.4619141, -33749.6523438, 32141.2324219
3: -7538.3774414, 20668.9472656, -6139.4248047, 16703.7578125, -24242.1347656, 26808.3710938
4: -24131.1914062, 14259.1669922, -19443.6015625, 11628.1406250, -35759.3320312, 33702.7578125

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7640212, upper bound: 19342.7692086
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7606479, upper bound: 19342.7613605
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -18757.4277344, 13746.7910156, -15552.2587891, 11533.6894531, -30291.1152344, 29299.0468750
1: -15182.3173828, 13317.3066406, -12591.0683594, 11168.5136719, -26350.8320312, 25908.3750000
2: -22137.7207031, 14540.8359375, -18365.9453125, 12196.6171875, -34334.3281250, 32906.7773438
3: -7595.2163086, 20821.6113281, -6361.1152344, 17325.3691406, -24920.5839844, 27182.7246094
4: -24311.6464844, 14362.3007812, -20172.7285156, 12046.0585938, -36357.7031250, 34535.0273438

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7640212, upper bound: 19342.7691344
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7606479, upper bound: 19342.7606479
time: 0.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.67 seconds
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8190363, upper bound: 19342.8674572
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8274170, upper bound: 19342.9172722
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8036989, upper bound: 19342.8623491
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8133079, upper bound: 19342.9101610
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8193230, upper bound: 19342.8646660
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8274282, upper bound: 19342.9144966
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8030434, upper bound: 19342.8596249
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.8122956, upper bound: 19342.9079102
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8344086
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8184497
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8345835
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8182567
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7640212, upper bound: 19342.7692086
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7606479, upper bound: 19342.7613605
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7640212, upper bound: 19342.7691344
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 3, lower bound: -19342.7606479, upper bound: 19342.7606479

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14240.0537109, 10552.5371094, -15756.9433594, 11699.3710938, -25939.4179688, 26309.4804688
1: -11525.4589844, 10218.2324219, -12754.5234375, 11328.3632812, -22853.8183594, 22972.7558594
2: -16815.0390625, 11163.8505859, -18604.3281250, 12373.6484375, -29188.6875000, 29768.1757812
3: -5824.7744141, 15837.9267578, -6454.3774414, 17558.2460938, -23383.0195312, 22292.3046875
4: -18475.3066406, 11033.8349609, -20437.7871094, 12222.4384766, -30697.7402344, 31471.6210938

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8190363, upper bound: 19342.8674572
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8190363, upper bound: 19342.8674572
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16168.9121094, 12118.5625000, -15608.5849609, 11602.2753906, -27771.1855469, 27727.1425781
1: -13087.1230469, 11729.5175781, -12633.0664062, 11234.7343750, -24321.8574219, 24362.5839844
2: -19143.3496094, 12796.8154297, -18435.2265625, 12271.8808594, -31415.2304688, 31232.0429688
3: -6637.7690430, 18063.5957031, -6399.1489258, 17405.5722656, -24043.3417969, 24462.7441406
4: -21031.5859375, 12630.6835938, -20252.2539062, 12120.5556641, -33152.1406250, 32882.9375000

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8274170, upper bound: 19342.9172722
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8274170, upper bound: 19342.9172722
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14185.5117188, 10511.4716797, -17838.9003906, 13232.8984375, -27418.4023438, 28350.3710938
1: -11481.2558594, 10178.7626953, -14442.9609375, 12814.6308594, -24295.8828125, 24621.7226562
2: -16751.1152344, 11120.6337891, -21065.7148438, 13996.4716797, -30747.5859375, 32186.3457031
3: -5801.3120117, 15777.3867188, -7303.9252930, 19891.6093750, -25692.9218750, 23081.3105469
4: -18404.8906250, 10990.7802734, -23144.0136719, 13835.2636719, -32240.1503906, 34134.7890625

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8036989, upper bound: 19342.8623491
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8036989, upper bound: 19342.8623491
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16124.2861328, 12084.4882812, -17691.8496094, 13140.9492188, -29265.2343750, 29776.3378906
1: -13050.9902344, 11696.9667969, -14322.1220703, 12725.8300781, -25776.8203125, 26019.0898438
2: -19091.2031250, 12761.0322266, -20900.5351562, 13899.7656250, -32990.9648438, 33661.5585938
3: -6618.1503906, 18013.9277344, -7251.9775391, 19744.1503906, -26362.3007812, 25265.9062500
4: -20973.9824219, 12594.8476562, -22960.9824219, 13738.4414062, -34712.4218750, 35555.8281250

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8133079, upper bound: 19342.9101610
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8133079, upper bound: 19342.9101610
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14810.9345703, 10955.6289062, -15969.5117188, 11850.9404297, -26661.8750000, 26925.1406250
1: -11988.1826172, 10610.2431641, -12926.7529297, 11475.5478516, -23463.7265625, 23536.9960938
2: -17489.0585938, 11591.2255859, -18855.4511719, 12533.2275391, -30022.2851562, 30446.6757812
3: -6051.2324219, 16469.9296875, -6538.7832031, 17794.0292969, -23845.2597656, 23008.7128906
4: -19216.7968750, 11455.9003906, -20714.9453125, 12380.8281250, -31597.6250000, 32170.8398438

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8193230, upper bound: 19342.8646660
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8193230, upper bound: 19342.8646660
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16757.7187500, 12539.3945312, -15817.8330078, 11751.8623047, -28509.5800781, 28357.2226562
1: -13563.3837891, 12137.8974609, -12802.5244141, 11380.0068359, -24943.3886719, 24940.4218750
2: -19836.9296875, 13242.3515625, -18682.7656250, 12429.3144531, -32266.2441406, 31925.1171875
3: -6873.6152344, 18715.0703125, -6482.4453125, 17638.1230469, -24511.7382812, 25197.5136719
4: -21795.2265625, 13071.4482422, -20525.4550781, 12276.8740234, -34072.1015625, 33596.8945312

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8274282, upper bound: 19342.9144966
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8274282, upper bound: 19342.9144966
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14772.0126953, 10926.9169922, -18030.1953125, 13374.7607422, -28146.7734375, 28957.1132812
1: -11956.6464844, 10582.4824219, -14597.7050781, 12952.2324219, -24908.8789062, 25180.1777344
2: -17443.6230469, 11561.0634766, -21292.3691406, 14146.2451172, -31589.8671875, 32853.4335938
3: -6035.1235352, 16427.1464844, -7383.5170898, 20106.2968750, -26141.4179688, 23810.6640625
4: -19166.7500000, 11425.9755859, -23394.4589844, 13983.5253906, -33150.2734375, 34820.4335938

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8030434, upper bound: 19342.8596249
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8030434, upper bound: 19342.8596249
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16726.8652344, 12516.3027344, -17879.7128906, 13280.7490234, -30007.6113281, 30396.0156250
1: -13538.4335938, 12115.8671875, -14473.9765625, 12861.4316406, -26399.8652344, 26589.8437500
2: -19801.2128906, 13218.2304688, -21123.2636719, 14047.3652344, -33848.5703125, 34341.4921875
3: -6860.6787109, 18681.2246094, -7330.3896484, 19955.4550781, -26816.1328125, 26011.6132812
4: -21755.7343750, 13047.2890625, -23207.0917969, 13884.4990234, -35640.2343750, 36254.3789062

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8122956, upper bound: 19342.9079102
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8122956, upper bound: 19342.9079102
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16806.6210938, 12360.7080078, -15117.2617188, 11236.7441406, -28043.3632812, 27477.9667969
1: -13604.6367188, 11970.4121094, -12237.2529297, 10877.6689453, -24482.3046875, 24207.6640625
2: -19853.5996094, 13072.4648438, -17851.5625000, 11880.9921875, -31734.5917969, 30924.0273438
3: -6835.0478516, 18672.0410156, -6195.7973633, 16849.3203125, -23684.3671875, 24867.8378906
4: -21809.7285156, 12916.4238281, -19609.8964844, 11734.9394531, -33544.6640625, 32526.3164062

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8344086
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8344086
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18861.4589844, 13874.5478516, -15069.4267578, 11200.8144531, -30062.2714844, 28943.9746094
1: -15271.3779297, 13435.1152344, -12198.6572266, 10843.0419922, -26114.4199219, 25633.7734375
2: -22281.5390625, 14671.8037109, -17795.7792969, 11842.6835938, -34124.2226562, 32467.5820312
3: -7670.6459961, 20974.7031250, -6174.8251953, 16796.4433594, -24467.0898438, 27149.5273438
4: -24479.0175781, 14505.2568359, -19548.3925781, 11696.8828125, -36175.8984375, 34053.6484375

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8184497
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8184497
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16990.9648438, 12485.7773438, -15671.0371094, 11631.3984375, -28622.3632812, 28156.8144531
1: -13754.2412109, 12092.0615234, -12685.5019531, 11261.9853516, -25016.2265625, 24777.5625000
2: -20071.0878906, 13205.1162109, -18505.9511719, 12298.6992188, -32369.7851562, 31711.0664062
3: -6907.4829102, 18873.6054688, -6414.8808594, 17463.7148438, -24371.1953125, 25288.4843750
4: -22049.8710938, 13047.8242188, -20330.1796875, 12148.2958984, -34198.1601562, 33378.0039062

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8345835
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8345835
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -19045.8925781, 14005.9316406, -15634.1699219, 11604.3466797, -30650.2343750, 29640.0976562
1: -15420.6894531, 13561.4599609, -12655.7441406, 11235.9130859, -26656.6015625, 26217.2031250
2: -22500.2187500, 14809.4160156, -18463.0078125, 12269.9960938, -34770.2109375, 33272.4218750
3: -7747.6997070, 21179.2539062, -6399.4897461, 17423.3066406, -25171.0058594, 27578.7441406
4: -24720.5703125, 14643.0927734, -20282.8183594, 12119.6992188, -36840.2695312, 34925.9101562

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8182567
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8182567
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18397.3164062, 13480.9345703, -14887.3261719, 11056.3623047, -29453.6777344, 28368.2617188
1: -14891.0351562, 13059.7236328, -12052.8779297, 10704.3808594, -25595.4160156, 25112.6015625
2: -21709.8203125, 14259.5947266, -17580.0488281, 11692.0224609, -33401.8359375, 31839.6445312
3: -7445.4697266, 20417.2187500, -6096.3774414, 16586.9296875, -24032.3984375, 26513.5957031
4: -23839.6992188, 14083.0869141, -19308.2148438, 11546.7861328, -35386.4843750, 33391.3007812

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7095382, upper bound: 19342.7353353
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7398484, upper bound: 19342.7450091
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20345.4687500, 14920.1904297, -14836.2031250, 11017.7001953, -31363.1679688, 29756.3906250
1: -16470.9492188, 14453.6923828, -12011.5859375, 10667.2724609, -27138.2187500, 26465.2773438
2: -24012.5410156, 15782.1562500, -17520.3632812, 11650.9375000, -35663.4726562, 33302.5117188
3: -8241.3447266, 22600.2148438, -6073.9379883, 16530.2070312, -24771.5488281, 28674.1503906
4: -26370.9746094, 15595.3349609, -19242.3769531, 11505.9658203, -37876.9375000, 34837.7109375

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7057285, upper bound: 19342.7274410
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7365537, upper bound: 19342.7373182
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18538.0488281, 13581.7792969, -15447.6396484, 11455.2861328, -29993.3320312, 29029.4160156
1: -15005.0410156, 13157.9501953, -12506.4873047, 11092.5039062, -26097.5410156, 25664.4375000
2: -21878.3320312, 14366.5068359, -18242.1406250, 12113.5839844, -33991.9140625, 32608.6484375
3: -7504.3217773, 20575.3671875, -6317.7436523, 17208.0332031, -24712.3496094, 26893.1113281
4: -24026.7734375, 14190.2373047, -20036.8203125, 11964.0917969, -35990.8554688, 34227.0507812

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7095382, upper bound: 19342.7349669
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7398484, upper bound: 19342.7444959
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20511.4062500, 15040.3310547, -15410.0039062, 11427.7099609, -31939.1152344, 30450.3339844
1: -16604.9101562, 14569.0214844, -12476.0722656, 11066.0830078, -27670.9902344, 27045.0917969
2: -24209.0566406, 15907.6572266, -18198.2363281, 12084.5058594, -36293.5625000, 34105.8945312
3: -8311.7041016, 22785.2656250, -6302.0751953, 17166.7187500, -25478.4218750, 29087.3378906
4: -26588.1992188, 15721.4736328, -19988.4042969, 11935.1064453, -38523.3046875, 35709.8789062

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7057285, upper bound: 19342.7270532
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7365537, upper bound: 19342.7365540
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.12 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8190363, upper bound: 19342.8674572
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8190363, upper bound: 19342.8674572
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8274170, upper bound: 19342.9172722
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8274170, upper bound: 19342.9172722
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8036989, upper bound: 19342.8623491
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8036989, upper bound: 19342.8623491
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8133079, upper bound: 19342.9101610
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8133079, upper bound: 19342.9101610
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8193230, upper bound: 19342.8646660
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8193230, upper bound: 19342.8646660
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8274282, upper bound: 19342.9144966
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8274282, upper bound: 19342.9144966
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8030434, upper bound: 19342.8596249
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8030434, upper bound: 19342.8596249
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8122956, upper bound: 19342.9079102
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.8122956, upper bound: 19342.9079102
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8344086
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8344086
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8184497
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8184497
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8345835
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7684629, upper bound: 19342.8345835
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8182567
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7649791, upper bound: 19342.8182567
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7095382, upper bound: 19342.7353353
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7398484, upper bound: 19342.7450091
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7057285, upper bound: 19342.7274410
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7365537, upper bound: 19342.7373182
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7095382, upper bound: 19342.7349669
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7398484, upper bound: 19342.7444959
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7057285, upper bound: 19342.7270532
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -19342.7365537, upper bound: 19342.7365540

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14240.0537109, 10552.5371094, -15625.8017578, 11602.8154297, -25842.8613281, 26178.3398438
1: -11525.4589844, 10218.2324219, -12648.3789062, 11235.2314453, -22760.6894531, 22866.6113281
2: -16815.0390625, 11163.8505859, -18448.6210938, 12271.4824219, -29086.5195312, 29612.4726562
3: -5824.7744141, 15837.9267578, -6400.9609375, 17411.5917969, -23236.3613281, 22238.8867188
4: -18475.3066406, 11033.8349609, -20266.8886719, 12121.1191406, -30596.4218750, 31300.7226562

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7642121, upper bound: 19342.8368309
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7639295, upper bound: 19342.7881034
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14240.0537109, 10552.5371094, -18177.6386719, 13434.1757812, -27674.2246094, 28730.1757812
1: -11525.4589844, 10218.2324219, -14713.9199219, 13006.8691406, -24532.3281250, 24932.1523438
2: -16815.0390625, 11163.8505859, -21472.5898438, 14207.3046875, -31022.3437500, 32636.4414062
3: -5824.7744141, 15837.9267578, -7413.5585938, 20244.0390625, -26068.8085938, 23251.4843750
4: -18475.3066406, 11033.8349609, -23581.2402344, 14028.2265625, -32503.5292969, 34615.0742188

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7642121, upper bound: 19342.8368309
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7639295, upper bound: 19342.7881034
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16168.9121094, 12118.5625000, -15480.1660156, 11507.7841797, -27676.6933594, 27598.7265625
1: -13087.1230469, 11729.5175781, -12529.0957031, 11143.6103516, -24230.7304688, 24258.6132812
2: -19143.3496094, 12796.8154297, -18282.7773438, 12171.8896484, -31315.2382812, 31079.5937500
3: -6637.7690430, 18063.5957031, -6346.8745117, 17261.9667969, -23899.7363281, 24410.4707031
4: -21031.5859375, 12630.6835938, -20084.9218750, 12021.4101562, -33052.9921875, 32715.6035156

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8078539, upper bound: 19342.8117957
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8243267, upper bound: 19342.9150348
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16168.9121094, 12118.5625000, -18034.8496094, 13341.0390625, -29509.9511719, 30153.4082031
1: -13087.1230469, 11729.5175781, -14596.9902344, 12916.7626953, -26003.8828125, 26326.5078125
2: -19143.3496094, 12796.8154297, -21309.5585938, 14109.4550781, -33252.8046875, 34106.3710938
3: -6637.7690430, 18063.5957031, -7360.5439453, 20096.6230469, -26734.3925781, 25424.1406250
4: -21031.5859375, 12630.6835938, -23402.3378906, 13930.1757812, -34961.7617188, 36033.0195312

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8078539, upper bound: 19342.8117957
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8243267, upper bound: 19342.9150348
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14185.5117188, 10511.4716797, -17706.1230469, 13134.8203125, -27320.3300781, 28217.5937500
1: -11481.2558594, 10178.7626953, -14335.4814453, 12720.3359375, -24201.5878906, 24514.2441406
2: -16751.1152344, 11120.6337891, -20907.9609375, 13893.0849609, -30644.1992188, 32028.5937500
3: -5801.3120117, 15777.3867188, -7249.8515625, 19742.9863281, -25544.2968750, 23027.2382812
4: -18404.8906250, 10990.7802734, -22970.9472656, 13732.7949219, -32137.6835938, 33961.7265625

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7519100, upper bound: 19342.8152986
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7520329, upper bound: 19342.7834984
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14185.5117188, 10511.4716797, -20179.8007812, 14897.5771484, -29083.0839844, 30691.2714844
1: -11481.2558594, 10178.7626953, -16338.3339844, 14427.3466797, -25908.5996094, 26517.0937500
2: -16751.1152344, 11120.6337891, -23839.0839844, 15759.4208984, -32510.5351562, 34959.7187500
3: -5801.3120117, 15777.3867188, -8227.9345703, 22481.9023438, -28283.2109375, 24005.3203125
4: -18404.8906250, 10990.7802734, -26183.5039062, 15571.7929688, -33976.6835938, 37174.2851562

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7519100, upper bound: 19342.8152986
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7520329, upper bound: 19342.7834984
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16124.2861328, 12084.4882812, -17560.2050781, 13043.7832031, -29168.0703125, 29644.6933594
1: -13050.9902344, 11696.9667969, -14215.5166016, 12632.4316406, -25683.4218750, 25912.4843750
2: -19091.2031250, 12761.0322266, -20744.0605469, 13797.3466797, -32888.5468750, 33505.0937500
3: -6618.1503906, 18013.9277344, -7198.4086914, 19596.8339844, -26214.9824219, 25212.3359375
4: -20973.9824219, 12594.8476562, -22789.3320312, 13636.9082031, -34610.8906250, 35384.1796875

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7911027, upper bound: 19342.8067968
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8105036, upper bound: 19342.9073043
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16124.2861328, 12084.4882812, -20038.5488281, 14808.9707031, -30933.2558594, 32123.0332031
1: -13050.9902344, 11696.9667969, -16222.4033203, 14341.7597656, -27392.7500000, 27919.3652344
2: -19091.2031250, 12761.0322266, -23680.5976562, 15666.1699219, -34757.3710938, 36441.6171875
3: -6618.1503906, 18013.9277344, -8177.8330078, 22339.8730469, -28958.0195312, 26191.7617188
4: -20973.9824219, 12594.8476562, -26007.9941406, 15478.5507812, -36452.5273438, 38602.8437500

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7911027, upper bound: 19342.8067968
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8105036, upper bound: 19342.9073043
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14810.9345703, 10955.6289062, -15837.4882812, 11753.7412109, -26564.6757812, 26793.1171875
1: -11988.1826172, 10610.2431641, -12819.8876953, 11381.7812500, -23369.9589844, 23430.1308594
2: -17489.0585938, 11591.2255859, -18698.7109375, 12430.3603516, -29919.4140625, 30289.9375000
3: -6051.2324219, 16469.9296875, -6484.9970703, 17646.3535156, -23697.5839844, 22954.9257812
4: -19216.7968750, 11455.9003906, -20542.9238281, 12278.8300781, -31495.6269531, 31998.8183594

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7632771, upper bound: 19342.8350362
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7602156, upper bound: 19342.7762322
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14810.9345703, 10955.6289062, -18367.8261719, 13565.3671875, -28376.3007812, 29323.4550781
1: -11988.1826172, 10610.2431641, -14868.3457031, 13136.2021484, -25124.3828125, 25478.5898438
2: -17489.0585938, 11591.2255859, -21697.6562500, 14346.8115234, -31835.8710938, 33288.8828125
3: -6051.2324219, 16469.9296875, -7488.7456055, 20453.6191406, -26504.8476562, 23958.6757812
4: -19216.7968750, 11455.9003906, -23830.0410156, 14166.7587891, -33383.5546875, 35285.9414062

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7632771, upper bound: 19342.8350362
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7602156, upper bound: 19342.7762322
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16757.7187500, 12539.3945312, -15688.6445312, 11656.8251953, -28414.5429688, 28228.0371094
1: -13563.3837891, 12137.8974609, -12697.9326172, 11288.3359375, -24851.7187500, 24835.8281250
2: -19836.9296875, 13242.3515625, -18529.3984375, 12328.7617188, -32165.6914062, 31771.7460938
3: -6873.6152344, 18715.0703125, -6429.8583984, 17493.6230469, -24367.2382812, 25144.9257812
4: -21795.2265625, 13071.4482422, -20357.1328125, 12177.1503906, -33972.3750000, 33428.5820312

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8082587, upper bound: 19342.8102713
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8243967, upper bound: 19342.9123566
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16757.7187500, 12539.3945312, -18220.8515625, 13469.8447266, -30227.5625000, 30760.2402344
1: -13563.3837891, 12137.8974609, -14748.0292969, 13043.6630859, -26607.0468750, 26885.9218750
2: -19836.9296875, 13242.3515625, -21530.1289062, 14246.4189453, -34083.3476562, 34772.4804688
3: -6873.6152344, 18715.0703125, -7434.3920898, 20302.2636719, -27175.8789062, 26149.4609375
4: -21795.2265625, 13071.4482422, -23646.2812500, 14066.1601562, -35861.3867188, 36717.7265625

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8082587, upper bound: 19342.8102713
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8243967, upper bound: 19342.9123566
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14772.0126953, 10926.9169922, -17895.9433594, 13275.6015625, -28047.6132812, 28822.8593750
1: -11956.6464844, 10582.4824219, -14489.0175781, 12856.9013672, -24813.5468750, 25071.4980469
2: -17443.6230469, 11561.0634766, -21132.8535156, 14041.7412109, -31485.3632812, 32693.9179688
3: -6035.1235352, 16427.1464844, -7328.8666992, 19956.0390625, -25991.1621094, 23756.0136719
4: -19166.7500000, 11425.9755859, -23219.4746094, 13879.9482422, -33046.6992188, 34645.4492188

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7510152, upper bound: 19342.8151144
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7496678, upper bound: 19342.7706989
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14772.0126953, 10926.9169922, -20382.2714844, 15047.2617188, -29819.2734375, 31309.1875000
1: -11956.6464844, 10582.4824219, -16501.9863281, 14572.4580078, -26529.1035156, 27084.4628906
2: -17443.6230469, 11561.0634766, -24078.7480469, 15917.5537109, -33361.1757812, 35639.8085938
3: -6035.1235352, 16427.1464844, -8312.1689453, 22708.4824219, -28743.6054688, 24739.3164062
4: -19166.7500000, 11425.9755859, -26448.2578125, 15728.4541016, -34895.2031250, 37874.2304688

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7510152, upper bound: 19342.8151144
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7496678, upper bound: 19342.7706989
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16726.8652344, 12516.3027344, -17746.2812500, 13182.2675781, -29909.1328125, 30262.5839844
1: -13538.4335938, 12115.8671875, -14365.9443359, 12766.7695312, -26305.2031250, 26481.8105469
2: -19801.2128906, 13218.2304688, -20964.6757812, 13943.5810547, -33744.7929688, 34182.9062500
3: -6860.6787109, 18681.2246094, -7276.1113281, 19806.1894531, -26666.8671875, 25957.3359375
4: -21755.7343750, 13047.2890625, -23033.1289062, 13781.6162109, -35537.3515625, 36080.4179688

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7905590, upper bound: 19342.8049197
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8096452, upper bound: 19342.9053783
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16726.8652344, 12516.3027344, -20234.9101562, 14954.6044922, -31681.4648438, 32751.2128906
1: -13538.4335938, 12115.8671875, -16381.1025391, 14482.9570312, -28021.3906250, 28496.9687500
2: -19801.2128906, 13218.2304688, -23913.4414062, 15820.0517578, -35621.2578125, 37131.6718750
3: -6860.6787109, 18681.2246094, -8259.8046875, 22560.1875000, -29420.8652344, 26941.0292969
4: -21755.7343750, 13047.2890625, -26264.9785156, 15630.9453125, -37386.6796875, 39312.2656250

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7905590, upper bound: 19342.8049197
time: 8.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8096452, upper bound: 19342.9053783
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16806.6210938, 12360.7080078, -14993.5380859, 11145.8652344, -27952.4824219, 27354.2460938
1: -13604.6367188, 11970.4121094, -12136.9570312, 10789.7783203, -24394.4101562, 24107.3632812
2: -19853.5996094, 13072.4648438, -17705.1777344, 11784.7763672, -31638.3750000, 30777.6425781
3: -6835.0478516, 18672.0410156, -6145.4360352, 16711.5195312, -23546.5664062, 24817.4746094
4: -21809.7285156, 12916.4238281, -19449.2656250, 11639.6513672, -33449.3750000, 32365.6855469

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7352336, upper bound: 19342.7770744
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7440950, upper bound: 19342.8147921
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16806.6210938, 12360.7080078, -16693.6523438, 12340.2441406, -29146.8652344, 29054.3593750
1: -13604.6367188, 11970.4121094, -13515.0771484, 11951.0302734, -25555.6679688, 25485.4824219
2: -19853.5996094, 13072.4648438, -19690.0488281, 13046.5390625, -32900.1367188, 32762.5136719
3: -6835.0478516, 18672.0410156, -6803.0429688, 18569.9531250, -25405.0000000, 25475.0839844
4: -21809.7285156, 12916.4238281, -21617.9355469, 12879.8779297, -34689.6054688, 34534.3593750

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7352336, upper bound: 19342.7770744
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7440950, upper bound: 19342.8147921
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -18861.4589844, 13874.5478516, -14945.9824219, 11110.2187500, -29971.6757812, 28820.5312500
1: -15271.3779297, 13435.1152344, -12098.5888672, 10755.3964844, -26026.7734375, 25533.7031250
2: -22281.5390625, 14671.8037109, -17649.7500000, 11746.7500000, -34028.2890625, 32321.5546875
3: -7670.6459961, 20974.7031250, -6124.6020508, 16658.9980469, -24329.6445312, 27099.3046875
4: -24479.0175781, 14505.2568359, -19388.1660156, 11601.8779297, -36080.8945312, 33893.4218750

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7640818
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8012057
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -18861.4589844, 13874.5478516, -16636.6210938, 12297.4560547, -31158.9121094, 30511.1679688
1: -15271.3779297, 13435.1152344, -13468.9541016, 11909.7294922, -27181.1074219, 26904.0703125
2: -22281.5390625, 14671.8037109, -19623.2070312, 13000.9892578, -35282.5273438, 34295.0117188
3: -7670.6459961, 20974.7031250, -6778.2500000, 18506.7480469, -26177.3945312, 27752.9511719
4: -24479.0175781, 14505.2568359, -21544.3222656, 12834.6396484, -37313.6562500, 36049.5703125

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7640818
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8012057
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16990.9648438, 12485.7773438, -15543.7558594, 11538.1250000, -28529.0898438, 28029.5332031
1: -13754.2412109, 12092.0615234, -12582.3281250, 11171.8496094, -24926.0898438, 24674.3906250
2: -20071.0878906, 13205.1162109, -18355.5390625, 12200.1406250, -32271.2265625, 31560.6562500
3: -6907.4829102, 18873.6054688, -6363.2919922, 17322.3105469, -24229.7929688, 25236.8984375
4: -22049.8710938, 13047.8242188, -20165.1679688, 12050.7001953, -34100.5625000, 33212.9882812

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7247944, upper bound: 19342.7484574
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7667574, upper bound: 19342.8338024
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16990.9648438, 12485.7773438, -17229.0722656, 12723.2216797, -29714.1875000, 29714.8496094
1: -13754.2412109, 12092.0615234, -13947.9560547, 12324.4394531, -26078.6796875, 26040.0175781
2: -20071.0878906, 13205.1162109, -20321.8691406, 13452.4697266, -33523.5546875, 33526.9843750
3: -6907.4829102, 18873.6054688, -7015.9780273, 19163.4609375, -26070.9414062, 25889.5839844
4: -22049.8710938, 13047.8242188, -22313.7910156, 13281.8193359, -35331.6914062, 35361.6093750

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7247944, upper bound: 19342.7484574
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7667574, upper bound: 19342.8338024
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -19045.8925781, 14005.9316406, -15507.0439453, 11511.2285156, -30557.1210938, 29512.9765625
1: -15420.6894531, 13561.4599609, -12552.6933594, 11145.9150391, -26566.6054688, 26114.1523438
2: -22500.2187500, 14809.4160156, -18312.7812500, 12171.5996094, -34671.8125000, 33122.1953125
3: -7747.6997070, 21179.2539062, -6347.9887695, 17282.0722656, -25029.7714844, 27527.2421875
4: -24720.5703125, 14643.0927734, -20118.0214844, 12022.2705078, -36742.8398438, 34761.1132812

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7639067
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8009847
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -19045.8925781, 14005.9316406, -17189.4687500, 12694.2890625, -31740.1777344, 31195.4003906
1: -15420.6894531, 13561.4599609, -13915.9150391, 12296.5419922, -27717.2304688, 27477.3730469
2: -22500.2187500, 14809.4160156, -20275.5664062, 13421.7685547, -35921.9843750, 35084.9804688
3: -7747.6997070, 21179.2539062, -6999.4975586, 19120.0117188, -26867.7109375, 28178.7519531
4: -24720.5703125, 14643.0927734, -22262.8066406, 13251.1982422, -37971.7695312, 36905.8984375

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7639067
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8009847
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -18397.3164062, 13480.9345703, -14587.5390625, 10834.8554688, -29232.1718750, 28068.4726562
1: -14891.0351562, 13059.7236328, -11811.3769531, 10492.4550781, -25383.4902344, 24871.1015625
2: -21709.8203125, 14259.5947266, -17230.7636719, 11456.5927734, -33166.4140625, 31490.3593750
3: -7445.4697266, 20417.2187500, -5973.6494141, 16259.9912109, -23705.4609375, 26390.8671875
4: -23839.6992188, 14083.0869141, -18922.0000000, 11312.5693359, -35152.2695312, 33005.0859375

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6978731, upper bound: 19342.7014530
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6978731, upper bound: 19342.7353353
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -18276.9531250, 13402.6904297, -16574.9570312, 12447.3632812, -30724.3164062, 29977.6484375
1: -14792.9208984, 12984.3300781, -13421.4658203, 12050.8291016, -26843.7500000, 26405.7949219
2: -21571.7109375, 14177.3740234, -19631.7890625, 13146.4130859, -34718.1210938, 33809.1601562
3: -7400.4428711, 20292.9335938, -6811.3652344, 18555.7539062, -25956.1972656, 27104.2988281
4: -23688.0917969, 14000.2656250, -21556.1679688, 12969.3828125, -36657.4726562, 35556.4335938

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7153531, upper bound: 19342.6424124
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7358307, upper bound: 19342.7405078
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -20345.4687500, 14920.1904297, -14536.6621094, 10796.2812500, -31141.7500000, 29456.8515625
1: -16470.9492188, 14453.6923828, -11770.2744141, 10455.3447266, -26926.2890625, 26223.9648438
2: -24012.5410156, 15782.1562500, -17171.3281250, 11415.4267578, -35427.9687500, 32953.4804688
3: -8241.3447266, 22600.2148438, -5951.2436523, 16203.4775391, -24444.8183594, 28551.4589844
4: -26370.9746094, 15595.3349609, -18856.4609375, 11271.9433594, -37642.9179688, 34451.7929688

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.6942250
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.7274410
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -20217.6445312, 14839.8066406, -16529.4179688, 12412.5771484, -32630.2167969, 31369.2246094
1: -16366.1884766, 14376.3261719, -13384.6191406, 12017.6552734, -28383.8437500, 27760.9453125
2: -23868.1386719, 15697.9794922, -19578.6992188, 13109.9101562, -36978.0429688, 35276.6796875
3: -8195.9355469, 22471.5917969, -6791.3994141, 18505.3828125, -26701.3183594, 29262.9921875
4: -26212.3828125, 15510.7099609, -21497.3906250, 12933.0195312, -39145.4023438, 37008.1015625

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7116177, upper bound: 19342.6377741
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7326730, upper bound: 19342.7333860
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -18538.0488281, 13581.7792969, -15146.4550781, 11233.1464844, -29771.1953125, 28728.2324219
1: -15005.0410156, 13157.9501953, -12263.8173828, 10879.8857422, -25884.9238281, 25421.7675781
2: -21878.3320312, 14366.5068359, -17891.1523438, 11877.5166016, -33755.8476562, 32257.6601562
3: -7504.3217773, 20575.3671875, -6194.8701172, 16879.7148438, -24384.0332031, 26770.2382812
4: -24026.7734375, 14190.2373047, -19648.8300781, 11729.7255859, -35756.4960938, 33839.0664062

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6714326, upper bound: 19342.6894886
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7083897, upper bound: 19342.7349460
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -18413.7988281, 13501.2783203, -17142.8984375, 12856.9931641, -31270.7929688, 30644.1757812
1: -14903.4257812, 13080.4130859, -13880.4570312, 12448.3417969, -27351.7656250, 26960.8710938
2: -21735.5000000, 14282.0488281, -20301.1484375, 13579.4228516, -35314.9179688, 34583.1953125
3: -7458.0991211, 20447.0996094, -7039.2167969, 19186.0156250, -26644.1132812, 27486.3144531
4: -23870.0273438, 14105.1171875, -22293.7089844, 13397.7900391, -37267.8164062, 36398.8281250

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6529391, upper bound: 19342.6803471
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7389978, upper bound: 19342.7438853
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -20511.4062500, 15040.3310547, -15109.8818359, 11206.1572266, -31717.5625000, 30150.2128906
1: -16604.9101562, 14569.0214844, -12234.2675781, 10854.0380859, -27458.9492188, 26803.2890625
2: -24209.0566406, 15907.6572266, -17848.5039062, 11849.2333984, -36058.2890625, 33756.1601562
3: -8311.7041016, 22785.2656250, -6179.5600586, 16839.5136719, -25151.2148438, 28964.8242188
4: -26588.1992188, 15721.4736328, -19601.8144531, 11701.5957031, -38289.7968750, 35323.2812500

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.6937951
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.7270532
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -20379.6816406, 14957.8515625, -17115.1308594, 12836.1210938, -33215.8046875, 32072.9765625
1: -16496.9394531, 14489.6113281, -13858.0224609, 12428.5820312, -28925.5175781, 28347.6328125
2: -24060.4414062, 15821.2460938, -20269.1425781, 13557.7822266, -37618.2226562, 36090.3906250
3: -8265.0781250, 22652.9921875, -7027.6152344, 19155.8398438, -27420.9179688, 29680.6054688
4: -26424.9628906, 15634.5888672, -22258.2500000, 13376.1083984, -39801.0703125, 37892.8359375

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7116177, upper bound: 19342.6373684
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7326730, upper bound: 19342.7326726
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.00 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7642121, upper bound: 19342.8368309
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7639295, upper bound: 19342.7881034
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7642121, upper bound: 19342.8368309
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7639295, upper bound: 19342.7881034
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8078539, upper bound: 19342.8117957
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8243267, upper bound: 19342.9150348
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8078539, upper bound: 19342.8117957
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8243267, upper bound: 19342.9150348
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7519100, upper bound: 19342.8152986
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7520329, upper bound: 19342.7834984
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7519100, upper bound: 19342.8152986
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7520329, upper bound: 19342.7834984
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7911027, upper bound: 19342.8067968
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8105036, upper bound: 19342.9073043
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7911027, upper bound: 19342.8067968
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8105036, upper bound: 19342.9073043
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7632771, upper bound: 19342.8350362
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7602156, upper bound: 19342.7762322
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7632771, upper bound: 19342.8350362
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7602156, upper bound: 19342.7762322
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8082587, upper bound: 19342.8102713
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8243967, upper bound: 19342.9123566
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8082587, upper bound: 19342.8102713
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8243967, upper bound: 19342.9123566
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7510152, upper bound: 19342.8151144
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7496678, upper bound: 19342.7706989
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7510152, upper bound: 19342.8151144
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7496678, upper bound: 19342.7706989
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7905590, upper bound: 19342.8049197
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8096452, upper bound: 19342.9053783
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7905590, upper bound: 19342.8049197
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.8096452, upper bound: 19342.9053783
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7352336, upper bound: 19342.7770744
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7440950, upper bound: 19342.8147921
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7352336, upper bound: 19342.7770744
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7440950, upper bound: 19342.8147921
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7640818
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8012057
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7640818
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8012057
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7247944, upper bound: 19342.7484574
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7667574, upper bound: 19342.8338024
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7247944, upper bound: 19342.7484574
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7667574, upper bound: 19342.8338024
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7639067
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8009847
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7309442, upper bound: 19342.7639067
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7405391, upper bound: 19342.8009847
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6978731, upper bound: 19342.7014530
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6978731, upper bound: 19342.7353353
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7153531, upper bound: 19342.6424124
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7358307, upper bound: 19342.7405078
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.6942250
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.7274410
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7116177, upper bound: 19342.6377741
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7326730, upper bound: 19342.7333860
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6714326, upper bound: 19342.6894886
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7083897, upper bound: 19342.7349460
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6529391, upper bound: 19342.6803471
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7389978, upper bound: 19342.7438853
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.6937951
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.6937951, upper bound: 19342.7270532
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7116177, upper bound: 19342.6373684
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 3, lower bound: -19342.7326730, upper bound: 19342.7326726

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13998.0400391, 10377.6494141, -15465.2626953, 11486.3046875, -25484.3437500, 25842.9082031
1: -11329.6835938, 10049.1777344, -12518.6416016, 11122.1806641, -22451.8632812, 22567.8164062
2: -16530.4414062, 10979.9091797, -18259.7246094, 12148.7216797, -28679.1640625, 29239.6328125
3: -5727.3505859, 15570.0966797, -6335.9804688, 17233.6992188, -22961.0488281, 21906.0781250
4: -18162.3593750, 10851.6083984, -20059.0468750, 11999.5937500, -30161.9511719, 30910.6523438

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7745448, upper bound: 19342.8476929
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7745448, upper bound: 19342.8476929
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13961.0722656, 10371.5205078, -15295.3203125, 11367.9765625, -25329.0429688, 25666.8378906
1: -11298.9882812, 10044.2363281, -12380.6933594, 11008.5576172, -22307.5468750, 22424.9277344
2: -16482.7890625, 10973.1484375, -18058.5605469, 12024.1718750, -28506.9609375, 29031.7089844
3: -5714.3222656, 15537.9033203, -6268.3950195, 17048.1210938, -22762.4433594, 21806.2988281
4: -18107.1660156, 10841.4804688, -19836.6953125, 11874.9785156, -29982.1386719, 30678.1757812

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7738428, upper bound: 19342.7861200
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7738428, upper bound: 19342.7947006
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13998.0400391, 10377.6494141, -18023.9101562, 13323.9062500, -27321.9453125, 28401.5585938
1: -11329.6835938, 10049.1777344, -14589.7871094, 12900.0917969, -24229.7753906, 24638.9648438
2: -16530.4414062, 10979.9091797, -21292.1210938, 14090.9501953, -30621.3886719, 32272.0312500
3: -5727.3505859, 15570.0966797, -7351.7153320, 20074.6933594, -25802.0429688, 22921.8125000
4: -18162.3593750, 10851.6083984, -23382.5410156, 13912.7480469, -32075.1035156, 34234.1484375

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7335144, upper bound: 19342.8282166
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7335144, upper bound: 19342.8368309
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13961.0722656, 10371.5205078, -17828.0234375, 13185.2324219, -27146.3046875, 28199.5429688
1: -11298.9882812, 10044.2363281, -14430.6835938, 12766.3105469, -24065.2988281, 24474.9160156
2: -16482.7890625, 10973.1484375, -21059.5781250, 13945.3017578, -30428.0898438, 32032.7265625
3: -5714.3222656, 15537.9033203, -7273.5551758, 19858.6914062, -25573.0136719, 22811.4589844
4: -18107.1660156, 10841.4804688, -23126.1855469, 13767.4423828, -31874.5996094, 33967.6640625

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7332912, upper bound: 19342.7788997
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7332912, upper bound: 19342.7788997
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15937.0537109, 11947.1962891, -15480.1660156, 11507.7841797, -27444.8378906, 27427.3632812
1: -12900.3271484, 11566.0097656, -12529.0957031, 11143.6103516, -24043.9335938, 24095.1054688
2: -18873.4472656, 12617.7890625, -18282.7773438, 12171.8896484, -31045.3359375, 30900.5664062
3: -6543.8129883, 17811.7656250, -6346.8745117, 17261.9667969, -23805.7792969, 24158.6406250
4: -20732.7128906, 12452.4453125, -20084.9218750, 12021.4101562, -32754.1191406, 32537.3671875

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6571761, upper bound: 19342.6625468
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9141653, upper bound: 19342.8222444
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -18436.2128906, 13926.4111328, -15363.6386719, 11435.9228516, -29872.1367188, 29243.7246094
1: -14922.7128906, 13487.3261719, -12434.0263672, 11074.4667969, -25997.1796875, 25899.5371094
2: -21877.6406250, 14702.6142578, -18152.3730469, 12096.7626953, -33974.4023438, 32836.3632812
3: -7588.9257812, 20694.4199219, -6307.0341797, 17145.1582031, -24719.2421875, 27001.4531250
4: -24021.7675781, 14488.9628906, -19939.3671875, 11946.0810547, -35967.8476562, 34414.2070312

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6700328, upper bound: 19342.7036058
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9274865, upper bound: 19342.9259394
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15937.0537109, 11947.1962891, -18034.8496094, 13341.0390625, -29278.0937500, 29982.0449219
1: -12900.3271484, 11566.0097656, -14596.9902344, 12916.7626953, -25817.0878906, 26163.0000000
2: -18873.4472656, 12617.7890625, -21309.5585938, 14109.4550781, -32982.8984375, 33927.3476562
3: -6543.8129883, 17811.7656250, -7360.5439453, 20096.6230469, -26640.4355469, 25172.3085938
4: -20732.7128906, 12452.4453125, -23402.3378906, 13930.1757812, -34662.8906250, 35854.7812500

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5891184, upper bound: 19342.5622505
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8073510, upper bound: 19342.8100666
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -18436.2128906, 13926.4111328, -17930.0898438, 13277.0537109, -31713.2656250, 31773.3984375
1: -14922.7128906, 13487.3261719, -14511.5693359, 12854.8867188, -27777.5996094, 27948.7109375
2: -21877.6406250, 14702.6142578, -21192.4667969, 14042.3916016, -35920.0312500, 35841.9765625
3: -7588.9257812, 20694.4199219, -7324.8115234, 19991.8710938, -27555.3632812, 28007.4824219
4: -24021.7675781, 14488.9628906, -23272.1484375, 13862.9687500, -37884.7343750, 37707.3164062

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5917571, upper bound: 19342.5687626
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8239416, upper bound: 19342.9132022
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13945.0039062, 10337.5302734, -17530.6933594, 13009.8789062, -26954.8828125, 27868.2226562
1: -11286.6884766, 10010.6201172, -14193.5585938, 12599.1455078, -23885.8339844, 24204.1777344
2: -16468.2304688, 10937.6679688, -20701.4472656, 13761.5400391, -30229.7695312, 31639.1152344
3: -5704.3681641, 15511.1064453, -7180.3676758, 19550.0390625, -25254.4062500, 22691.4707031
4: -18093.8261719, 10809.5253906, -22743.9355469, 13602.5322266, -31696.3593750, 33553.4570312

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7628058, upper bound: 19342.8346208
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7628058, upper bound: 19342.8346208
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13914.8603516, 10336.7500000, -17393.2695312, 12908.7548828, -26823.6152344, 27730.0195312
1: -11261.5322266, 10010.8779297, -14081.9921875, 12502.0781250, -23763.6074219, 24092.8710938
2: -16428.7304688, 10936.5058594, -20538.2812500, 13655.0332031, -30083.7636719, 31474.7871094
3: -5694.2797852, 15486.8291016, -7122.4267578, 19395.6093750, -25089.8886719, 22609.2519531
4: -18047.6132812, 10804.9433594, -22563.2910156, 13496.4228516, -31544.0351562, 33368.2304688

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7623910, upper bound: 19342.7811474
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7623910, upper bound: 19342.7811474
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13945.0039062, 10337.5302734, -20013.2441406, 14779.6728516, -28724.6757812, 30350.7714844
1: -11286.6884766, 10010.6201172, -16203.3544922, 14313.0810547, -25599.7695312, 26213.9707031
2: -16468.2304688, 10937.6679688, -23643.4414062, 15635.0361328, -32103.2636719, 34581.1093750
3: -5704.3681641, 15511.1064453, -8161.9785156, 22298.8417969, -28003.2109375, 23673.0859375
4: -18093.8261719, 10809.5253906, -25968.1835938, 15448.3339844, -33542.1562500, 36777.7031250

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226545, upper bound: 19342.8068873
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226545, upper bound: 19342.8068873
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13914.8603516, 10336.7500000, -19848.4355469, 14658.0566406, -28572.9179688, 30185.1855469
1: -11261.5322266, 10010.8779297, -16069.9951172, 14196.0585938, -25457.5859375, 26080.8710938
2: -16428.7304688, 10936.5058594, -23448.0371094, 15507.6005859, -31936.3320312, 34384.5312500
3: -5694.2797852, 15486.8291016, -8093.0854492, 22114.4121094, -27808.6914062, 23579.9121094
4: -18047.6132812, 10804.9433594, -25752.5878906, 15321.5537109, -33369.1640625, 36557.5312500

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7230201, upper bound: 19342.7739160
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7230201, upper bound: 19342.7834984
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15892.4570312, 11913.3593750, -17560.2050781, 13043.7832031, -28936.2382812, 29473.5644531
1: -12864.1494141, 11533.7285156, -14215.5166016, 12632.4316406, -25496.5820312, 25749.2460938
2: -18821.2265625, 12582.2773438, -20744.0605469, 13797.3466797, -32618.5742188, 33326.3359375
3: -6524.4311523, 17762.1269531, -7198.4086914, 19596.8339844, -26121.2656250, 24960.5351562
4: -20675.0917969, 12416.9257812, -22789.3320312, 13636.9082031, -34312.0000000, 35206.2578125

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8057594, upper bound: 19342.8056740
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8057594, upper bound: 19342.8056740
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -18393.5039062, 13893.5195312, -17451.3984375, 12977.9003906, -31371.4042969, 31276.7871094
1: -14888.2060547, 13455.9375000, -14126.3808594, 12569.0576172, -27457.2636719, 27540.3398438
2: -21828.2539062, 14668.1650391, -20622.4980469, 13728.5107422, -35556.7500000, 35239.0937500
3: -7570.2011719, 20647.1621094, -7162.0683594, 19489.3496094, -27020.8125000, 27790.7343750
4: -23966.9726562, 14454.5478516, -22653.9042969, 13567.9033203, -37534.8710938, 37062.2382812

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8798180, upper bound: 19342.8118053
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7952321, upper bound: 19342.8084736
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15892.4570312, 11913.3593750, -20038.5488281, 14808.9707031, -30701.4238281, 31951.9082031
1: -12864.1494141, 11533.7285156, -16222.4033203, 14341.7597656, -27205.9101562, 27756.1289062
2: -18821.2265625, 12582.2773438, -23680.5976562, 15666.1699219, -34487.3945312, 36262.8710938
3: -6524.4311523, 17762.1269531, -8177.8330078, 22339.8730469, -28864.3027344, 25939.9609375
4: -20675.0917969, 12416.9257812, -26007.9941406, 15478.5507812, -36153.6406250, 38424.9140625

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6937972, upper bound: 19342.7922481
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6937972, upper bound: 19342.8067968
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -18393.5039062, 13893.5195312, -19940.1953125, 14751.1601562, -33144.6640625, 33728.0585938
1: -14888.2060547, 13455.9375000, -16141.8886719, 14286.1210938, -29174.3281250, 29526.1972656
2: -21828.2539062, 14668.1650391, -23571.5703125, 15605.7080078, -37433.9531250, 38150.2382812
3: -7570.2011719, 20647.1621094, -8145.9560547, 22243.9804688, -29763.5488281, 28759.1523438
4: -23966.9726562, 14454.5478516, -25886.4960938, 15417.8349609, -39384.8046875, 40252.5156250

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7979555, upper bound: 19342.8932321
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7422072, upper bound: 19342.7718159
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14579.2099609, 10787.3642578, -15682.7714844, 11641.5449219, -26220.7539062, 26470.1367188
1: -11800.7890625, 10447.4912109, -12694.8300781, 11273.0615234, -23073.8515625, 23142.3203125
2: -17216.6347656, 11414.1083984, -18516.5195312, 12312.2089844, -29528.8437500, 29930.6289062
3: -5957.5449219, 16213.3701172, -6422.4931641, 17474.8339844, -23432.3789062, 22635.8613281
4: -18917.0625000, 11280.6767578, -20342.5214844, 12161.8281250, -31078.8906250, 31623.1972656

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7736832, upper bound: 19342.8433285
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7736832, upper bound: 19342.8433285
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14478.3515625, 10742.5556641, -15496.2431641, 11510.8867188, -25989.2382812, 26238.7988281
1: -11717.9423828, 10404.5957031, -12543.5078125, 11147.3134766, -22865.2558594, 22948.1035156
2: -17095.2695312, 11365.3291016, -18296.0371094, 12174.5000000, -29269.7695312, 29661.3671875
3: -5921.4726562, 16111.9628906, -6347.9814453, 17270.4511719, -23191.9238281, 22459.9414062
4: -18781.0488281, 11231.2031250, -20098.8964844, 12024.4853516, -30805.5292969, 31330.0976562

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7709032, upper bound: 19342.7750364
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7709032, upper bound: 19342.7827582
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14579.2099609, 10787.3642578, -18218.7363281, 13458.0732422, -28037.2832031, 29006.0996094
1: -11800.7890625, 10447.4912109, -14747.7871094, 13032.2519531, -24833.0410156, 25195.2773438
2: -17216.6347656, 11414.1083984, -21522.2968750, 14233.5566406, -31450.1914062, 32936.3984375
3: -5957.5449219, 16213.3701172, -7428.5698242, 20288.9316406, -26246.4765625, 23641.9375000
4: -18917.0625000, 11280.6767578, -23636.9550781, 14054.3535156, -32971.4140625, 34917.6328125

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7325806, upper bound: 19342.8265641
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7325806, upper bound: 19342.8350362
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14478.3515625, 10742.5556641, -18006.2500000, 13308.4580078, -27786.8085938, 28748.8046875
1: -11717.9423828, 10404.5957031, -14575.4687500, 12887.5146484, -24605.4531250, 24980.0585938
2: -17095.2695312, 11365.3291016, -21271.2285156, 14076.1728516, -31171.4414062, 32636.5585938
3: -5921.4726562, 16111.9628906, -7344.1831055, 20055.4628906, -25976.9355469, 23456.1445312
4: -18781.0488281, 11231.2031250, -23360.3183594, 13897.5283203, -32678.5761719, 34591.5117188

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7299120, upper bound: 19342.7679147
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7299120, upper bound: 19342.7679147
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16525.1894531, 12368.0322266, -15688.6445312, 11656.8251953, -28182.0156250, 28056.6757812
1: -13376.2695312, 11974.1875000, -12697.9326172, 11288.3359375, -24664.6054688, 24672.1210938
2: -19566.5722656, 13063.0722656, -18529.3984375, 12328.7617188, -31895.3339844, 31592.4707031
3: -6779.5102539, 18462.9707031, -6429.8583984, 17493.6230469, -24273.1328125, 24892.8261719
4: -21495.9433594, 12893.1914062, -20357.1328125, 12177.1503906, -33673.0859375, 33250.3242188

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6998593, upper bound: 19342.7099346
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9116115, upper bound: 19342.8204912
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19016.1386719, 14344.3164062, -15569.5498047, 11583.4599609, -30599.5976562, 29874.0234375
1: -15391.2978516, 13891.4980469, -12600.7011719, 11217.7783203, -26609.0703125, 26476.0644531
2: -22558.7675781, 15143.4609375, -18396.1582031, 12251.9814453, -34810.7500000, 33528.2617188
3: -7822.5371094, 21335.5703125, -6389.1401367, 17374.4082031, -25183.6699219, 27724.7089844
4: -24772.8183594, 14926.4414062, -20208.3125000, 12100.2275391, -36873.0390625, 35127.6289062

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7753824, upper bound: 19342.8009372
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9251333, upper bound: 19342.9231951
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16525.1894531, 12368.0322266, -18220.8515625, 13469.8447266, -29995.0351562, 30588.8789062
1: -13376.2695312, 11974.1875000, -14748.0292969, 13043.6630859, -26419.9316406, 26722.2167969
2: -19566.5722656, 13063.0722656, -21530.1289062, 14246.4189453, -33812.9921875, 34593.2031250
3: -6779.5102539, 18462.9707031, -7434.3920898, 20302.2636719, -27081.7734375, 25897.3593750
4: -21495.9433594, 12893.1914062, -23646.2812500, 14066.1601562, -35562.1015625, 36539.4726562

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5975124, upper bound: 19342.6623757
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8075630, upper bound: 19342.8087818
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19016.1386719, 14344.3164062, -18113.3281250, 13404.3779297, -32420.5156250, 32381.6113281
1: -15391.2978516, 13891.4980469, -14660.2939453, 12980.2382812, -28371.5332031, 28507.1875000
2: -22558.7675781, 15143.4609375, -21410.1621094, 14177.7119141, -36736.4726562, 36507.3281250
3: -7822.5371094, 21335.5703125, -7397.8168945, 20195.0019531, -27993.3496094, 28723.9257812
4: -24772.8183594, 14926.4414062, -23512.6972656, 13997.3037109, -38770.1171875, 38392.0156250

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7418834, upper bound: 19342.7750444
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8239399, upper bound: 19342.9107566
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14543.2070312, 10760.9658203, -17736.3886719, 13160.0859375, -27703.2929688, 28497.3515625
1: -11771.5820312, 10421.9804688, -14360.1074219, 12745.0078125, -24516.5878906, 24782.0839844
2: -17174.5957031, 11386.4072266, -20945.2187500, 13920.2773438, -31094.8730469, 32331.6250000
3: -5942.7397461, 16173.8945312, -7264.6054688, 19779.5078125, -25722.2460938, 23438.5000000
4: -18870.7851562, 11253.1855469, -23013.1367188, 13759.7167969, -32630.5019531, 34266.3203125

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7619380, upper bound: 19342.8285332
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7619380, upper bound: 19342.8285332
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14439.7343750, 10714.0156250, -17554.7832031, 13033.5400391, -27473.2734375, 28268.7988281
1: -11686.6962891, 10377.1728516, -14212.5136719, 12622.7724609, -24309.4687500, 24589.6855469
2: -17050.1738281, 11335.2919922, -20730.0312500, 13786.7119141, -30836.8867188, 32065.3222656
3: -5905.4018555, 16069.5273438, -7192.5346680, 19580.0117188, -25485.4140625, 23262.0605469
4: -18731.3554688, 11201.1904297, -22775.2988281, 13626.6083984, -32357.9628906, 33976.4882812

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7598590, upper bound: 19342.7694414
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7598590, upper bound: 19342.7694414
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14543.2070312, 10760.9658203, -20230.9492188, 14938.2070312, -29481.4140625, 30991.9140625
1: -11771.5820312, 10421.9804688, -16379.6591797, 14466.9492188, -26238.5312500, 26801.6367188
2: -17174.5957031, 11386.4072266, -23900.8242188, 15802.7031250, -32977.2968750, 35287.2304688
3: -5942.7397461, 16173.8945312, -8251.1542969, 22541.2695312, -28484.0078125, 24425.0488281
4: -18870.7851562, 11253.1855469, -26252.3164062, 15614.4833984, -34485.2656250, 37505.5000000

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7217170, upper bound: 19342.8068008
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7217170, upper bound: 19342.8068008
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14439.7343750, 10714.0156250, -20020.8183594, 14790.7597656, -29230.4902344, 30734.8320312
1: -11686.6962891, 10377.1728516, -16209.2519531, 14324.3251953, -26011.0214844, 26586.4257812
2: -17050.1738281, 11335.2919922, -23652.4628906, 15647.5732422, -32697.7460938, 34987.7539062
3: -5905.4018555, 16069.5273438, -8167.8242188, 22310.4042969, -28215.8066406, 24237.3515625
4: -18731.3554688, 11201.1904297, -25978.5175781, 15460.1035156, -34191.4609375, 37179.6992188

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7207377, upper bound: 19342.7622697
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7207377, upper bound: 19342.7622697
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16494.3457031, 12345.0732422, -17746.2812500, 13182.2675781, -29676.6113281, 30091.3554688
1: -13351.2900391, 11952.2978516, -14365.9443359, 12766.7695312, -26118.0566406, 26318.2421875
2: -19530.8046875, 13039.0986328, -20964.6757812, 13943.5810547, -33474.3867188, 34003.7734375
3: -6766.7177734, 18429.1230469, -7276.1113281, 19806.1894531, -26572.9062500, 25705.2343750
4: -21456.4238281, 12869.2031250, -23033.1289062, 13781.6162109, -35238.0390625, 35902.3320312

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8039755, upper bound: 19342.8039755
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8039755, upper bound: 19342.8177843
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -18986.0859375, 14321.7500000, -17636.0761719, 13115.6259766, -32101.7109375, 31897.8593750
1: -15367.0537109, 13870.0292969, -14275.6064453, 12702.6669922, -28069.7207031, 28110.8457031
2: -22524.2832031, 15120.0693359, -20841.4765625, 13873.9443359, -36398.2265625, 35919.0429688
3: -7810.0727539, 21302.8242188, -7239.3422852, 19697.4257812, -27471.4550781, 28526.2890625
4: -24734.5625000, 14902.9335938, -22895.9140625, 13711.8144531, -38446.3750000, 37761.6328125

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9031589, upper bound: 19342.9000526
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8998628, upper bound: 19342.8998621
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16494.3457031, 12345.0732422, -20234.9101562, 14954.6044922, -31448.9414062, 32579.9824219
1: -13351.2900391, 11952.2978516, -16381.1025391, 14482.9570312, -27834.2441406, 28333.4003906
2: -19530.8046875, 13039.0986328, -23913.4414062, 15820.0517578, -35350.8554688, 36952.5390625
3: -6766.7177734, 18429.1230469, -8259.8046875, 22560.1875000, -29326.9003906, 26688.9277344
4: -21456.4238281, 12869.2031250, -26264.9785156, 15630.9453125, -37087.3671875, 39134.1796875

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6929748, upper bound: 19342.7905013
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6929748, upper bound: 19342.8049197
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -18986.0859375, 14321.7500000, -20135.1093750, 14895.9863281, -33882.0703125, 34358.7148438
1: -15367.0537109, 13870.0292969, -16299.3837891, 14426.5332031, -29793.5859375, 30104.4960938
2: -22524.2832031, 15120.0693359, -23802.7050781, 15758.6855469, -38282.9648438, 38841.4296875
3: -7810.0727539, 21302.8242188, -8227.4707031, 22462.9121094, -30224.6992188, 29498.8828125
4: -24734.5625000, 14902.9335938, -26141.6015625, 15569.3691406, -40303.9296875, 40964.1054688

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7922199, upper bound: 19342.8836937
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7420350, upper bound: 19342.8581591
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16507.3007812, 12135.6513672, -14993.5380859, 11145.8652344, -27653.1640625, 27129.1894531
1: -13363.3476562, 11755.5947266, -12136.9570312, 10789.7783203, -24153.1191406, 23892.5468750
2: -19504.2890625, 12837.2812500, -17705.1777344, 11784.7763672, -31289.0644531, 30542.4589844
3: -6712.8637695, 18345.2851562, -6145.4360352, 16711.5195312, -23424.3828125, 24490.7187500
4: -21423.7128906, 12683.0146484, -19449.2656250, 11639.6513672, -33063.3632812, 32132.2812500

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8004381, upper bound: 19342.7815318
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8004381, upper bound: 19342.7815318
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18465.5917969, 13732.3798828, -14843.0673828, 11046.9316406, -29512.5234375, 28575.4453125
1: -14949.3984375, 13294.2597656, -12013.9121094, 10694.2285156, -25643.6269531, 25308.1718750
2: -21865.9062500, 14499.8681641, -17533.4980469, 11680.8906250, -33546.7968750, 32033.3632812
3: -7538.8730469, 20604.2441406, -6089.1811523, 16556.4335938, -24095.3046875, 26693.4257812
4: -24016.7031250, 14309.7402344, -19260.8925781, 11535.5644531, -35552.2656250, 33570.6328125

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8097509, upper bound: 19342.8192192
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8097509, upper bound: 19342.8192192
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16507.3007812, 12135.6513672, -16693.6523438, 12340.2441406, -28847.5449219, 28829.3046875
1: -13363.3476562, 11755.5947266, -13515.0771484, 11951.0302734, -25314.3750000, 25270.6699219
2: -19504.2890625, 12837.2812500, -19690.0488281, 13046.5390625, -32550.8281250, 32527.3300781
3: -6712.8637695, 18345.2851562, -6803.0429688, 18569.9531250, -25282.8164062, 25148.3281250
4: -21423.7128906, 12683.0146484, -21617.9355469, 12879.8779297, -34303.5898438, 34300.9492188

Time for backsubstitution: 1.97 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.11 + 417.52 = 421.63 seconds
