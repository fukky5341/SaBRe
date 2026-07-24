## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 8119.054186346224


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250)
1: (-2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836)
2: (-4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766)
3: (-5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625)
4: (-4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.58 + 2.30 = 5.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -8119.1353780, upper bound: 8119.1353774

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1353652, upper bound: 8119.1353393
time: 0.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1353399, upper bound: 8119.1353393
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.84 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 3, lower bound: -8119.1353652, upper bound: 8119.1353393
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 3, lower bound: -8119.1353399, upper bound: 8119.1353393

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -25404.3964844, 27856.8964844, -25754.6015625, 28227.9238281, -53632.3203125, 53611.4843750
1: -2787.2885742, 1947.0169678, -2824.3879395, 1974.0302734, -4761.3188477, 4771.4047852
2: -4440.2929688, 5185.1235352, -4501.1704102, 5254.6303711, -9694.9238281, 9686.2929688
3: -5075.7934570, 3278.2241211, -5145.2661133, 3322.6293945, -8398.4228516, 8423.4882812
4: -3830.7941895, 4209.2246094, -3883.3461914, 4265.2788086, -8096.0727539, 8092.5708008

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349760, upper bound: 8119.1351817
time: 0.81 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349758, upper bound: 8119.1349751
time: 0.89 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -26407.0703125, 29049.6074219, -27150.1699219, 29843.3476562, -56250.4179688, 56199.7773438
1: -2904.0537109, 2023.8817139, -2984.1164551, 2080.9199219, -4984.9736328, 5007.9980469
2: -4620.1679688, 5406.9750977, -4749.4809570, 5554.3808594, -10174.5488281, 10156.4560547
3: -5286.1210938, 3414.1989746, -5433.3447266, 3508.6970215, -8794.8183594, 8847.5439453
4: -3985.5661621, 4391.5874023, -4097.6621094, 4510.5859375, -8496.1523438, 8489.2480469

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349759, upper bound: 8119.1352331
time: 0.86 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349757, upper bound: 8119.1349750
time: 0.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.92 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 3, lower bound: -8119.1349760, upper bound: 8119.1351817
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 3, lower bound: -8119.1349758, upper bound: 8119.1349751
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 3, lower bound: -8119.1349759, upper bound: 8119.1352331
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.92
Output dim: 3, lower bound: -8119.1349757, upper bound: 8119.1349750

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -25158.1796875, 27599.7656250, -25337.7519531, 27793.1289062, -52951.3046875, 52937.5156250
1: -2761.1008301, 1928.0463867, -2780.0920410, 1941.9372559, -4703.0380859, 4708.1386719
2: -4396.9628906, 5136.3525391, -4427.7963867, 5172.1459961, -9569.1074219, 9564.1474609
3: -5025.0844727, 3247.3625488, -5059.4072266, 3270.4084473, -8295.4921875, 8306.7695312
4: -3792.4257812, 4170.2465820, -3818.3457031, 4199.3305664, -7991.7563477, 7988.5913086

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346688, upper bound: 8119.1346937
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331408, upper bound: 8119.1340103
time: 0.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -25151.4238281, 27583.3886719, -29966.5468750, 32985.1875000, -58136.6093750, 57549.9375000
1: -2759.6457520, 1927.4000244, -3297.0676270, 2295.2238770, -5054.8696289, 5224.4677734
2: -4395.8925781, 5134.1386719, -5241.9033203, 6139.3583984, -10535.2509766, 10376.0400391
3: -5024.6303711, 3245.9731445, -5988.5063477, 3875.5080566, -8900.1376953, 9234.4794922
4: -3791.9970703, 4168.0727539, -4516.8500977, 4984.2690430, -8776.2646484, 8684.9228516

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346648, upper bound: 8119.1346143
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331411, upper bound: 8119.1339017
time: 0.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -26153.8847656, 28784.2734375, -26709.1289062, 29381.1250000, -55535.0078125, 55493.4023438
1: -2877.0278320, 2004.3865967, -2937.0412598, 2046.9122314, -4923.9399414, 4941.4277344
2: -4575.5219727, 5356.6684570, -4671.8515625, 5466.8105469, -10042.3320312, 10028.5195312
3: -5233.7622070, 3382.2758789, -5342.4414062, 3453.0988770, -8686.8613281, 8724.7119141
4: -3945.9816895, 4351.1948242, -4028.8913574, 4440.4638672, -8386.4453125, 8380.0859375

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347267, upper bound: 8119.1351561
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1348159, upper bound: 8119.1351667
time: 0.81 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -26145.5917969, 28776.8476562, -30074.0312500, 33057.9218750, -59203.5156250, 58850.8789062
1: -2876.2475586, 2003.6612549, -3305.2187500, 2305.0617676, -5181.3095703, 5308.8793945
2: -4574.5449219, 5355.5712891, -5261.9882812, 6153.0659180, -10727.6093750, 10617.5585938
3: -5233.4394531, 3381.4870605, -6017.5639648, 3886.5283203, -9119.9648438, 9399.0498047
4: -3945.4235840, 4350.2202148, -4539.5805664, 4994.9487305, -8940.3720703, 8889.7988281

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347265, upper bound: 8119.1348159
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1348156, upper bound: 8119.1348149
time: 1.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.55 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1346688, upper bound: 8119.1346937
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1331408, upper bound: 8119.1340103
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1346648, upper bound: 8119.1346143
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1331411, upper bound: 8119.1339017
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1347267, upper bound: 8119.1351561
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1348159, upper bound: 8119.1351667
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1347265, upper bound: 8119.1348159
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.55
Output dim: 3, lower bound: -8119.1348156, upper bound: 8119.1348149

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -24417.2089844, 26773.9941406, -24728.3886719, 27111.5917969, -51528.7968750, 51502.3828125
1: -2679.3474121, 1870.7454834, -2713.2187500, 1894.9218750, -4574.2695312, 4583.9643555
2: -4267.2250977, 4983.1528320, -4321.3037109, 5046.4658203, -9313.6914062, 9304.4560547
3: -4876.4477539, 3150.7856445, -4938.4663086, 3190.7265625, -8067.1743164, 8089.2519531
4: -3680.2065430, 4046.1757812, -3727.3181152, 4097.1054688, -7777.3120117, 7773.4941406

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345237, upper bound: 8119.1341401
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345371, upper bound: 8119.1345868
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -24229.8105469, 26492.7441406, -25117.9824219, 27518.8378906, -51748.6484375, 51610.7265625
1: -2652.2219238, 1857.1925049, -2753.5424805, 1924.6851807, -4576.9067383, 4610.7343750
2: -4232.1298828, 4937.3120117, -4389.9462891, 5125.0927734, -9357.2197266, 9327.2578125
3: -4838.1923828, 3121.7380371, -5019.1655273, 3239.8950195, -8078.0874023, 8140.9033203
4: -3652.3334961, 4006.5068359, -3787.7338867, 4160.3637695, -7812.6972656, 7794.2397461

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330131, upper bound: 8119.1329789
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330818, upper bound: 8119.1339025
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -24352.0273438, 26686.7148438, -29512.6230469, 32471.3652344, -56823.3906250, 56199.3359375
1: -2671.0273438, 1865.5965576, -3246.2329102, 2260.3493652, -4931.3769531, 5111.8295898
2: -4255.8803711, 4967.9384766, -5161.8237305, 6045.1406250, -10301.0214844, 10129.7617188
3: -4864.4638672, 3141.3293457, -5897.6230469, 3816.0693359, -8680.5322266, 9038.9521484
4: -3671.2558594, 4033.2182617, -4448.0595703, 4907.2348633, -8578.4902344, 8481.2773438

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345202, upper bound: 8119.1340293
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345339, upper bound: 8119.1345046
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -24205.5058594, 26462.4121094, -29267.0546875, 32099.6210938, -56305.1250000, 55729.4687500
1: -2649.3205566, 1855.1752930, -3211.9858398, 2241.8918457, -4891.2124023, 5067.1611328
2: -4228.1870117, 4932.0913086, -5116.6557617, 5981.4667969, -10209.6533203, 10048.7470703
3: -4834.4331055, 3118.4860840, -5850.5869141, 3777.3823242, -8611.8154297, 8969.0712891
4: -3649.2717285, 4002.1308594, -4413.8554688, 4853.3339844, -8502.6054688, 8415.9863281

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330270, upper bound: 8119.1328248
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330819, upper bound: 8119.1337999
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -25362.9531250, 27965.7304688, -26289.9628906, 28948.9101562, -54311.8632812, 54255.6953125
1: -2793.5688477, 1944.4047852, -2892.9250488, 2015.0837402, -4808.6523438, 4837.3300781
2: -4437.6040039, 5203.1333008, -4598.6513672, 5385.6894531, -9823.2929688, 9801.7851562
3: -5074.0996094, 3284.1325684, -5257.9252930, 3401.1694336, -8475.2695312, 8542.0566406
4: -3824.9929199, 4226.9990234, -3964.7663574, 4374.8559570, -8199.8486328, 8191.7656250

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345652, upper bound: 8119.1350564
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345654, upper bound: 8119.1349614
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -28354.2011719, 31471.7441406, -26575.2304688, 29244.1835938, -57598.3828125, 58046.9726562
1: -3137.0500488, 2166.7109375, -2922.9567871, 2036.6596680, -5173.7089844, 5089.6679688
2: -4955.6806641, 5855.5288086, -4648.4663086, 5440.8198242, -10396.4990234, 10503.9951172
3: -5661.8842773, 3690.2915039, -5314.9736328, 3436.5598145, -9098.4443359, 9005.2636719
4: -4259.7846680, 4759.8896484, -4008.0500488, 4419.4526367, -8679.2373047, 8767.9394531

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347155, upper bound: 8119.1350593
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347151, upper bound: 8119.1349860
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -25357.6679688, 27961.8261719, -29643.7226562, 32613.2460938, -57970.9140625, 57605.5468750
1: -2793.1333008, 1943.9172363, -3259.8730469, 2272.4169922, -5065.5493164, 5203.7900391
2: -4437.1181641, 5202.7221680, -5186.7270508, 6069.7294922, -10506.8447266, 10389.4482422
3: -5074.3750000, 3283.7731934, -5930.4101562, 3833.1921387, -8907.5673828, 9214.1835938
4: -3824.8457031, 4226.7006836, -4473.4594727, 4927.5400391, -8752.3837891, 8700.1601562

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345652, upper bound: 8119.1347662
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345654, upper bound: 8119.1347141
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -28338.8535156, 31456.5507812, -29978.8398438, 32958.6132812, -61297.4648438, 61435.3906250
1: -3135.4521484, 2165.4448242, -3295.0581055, 2297.7556152, -5433.2080078, 5460.5029297
2: -4953.5034180, 5852.9506836, -5245.3569336, 6134.3090820, -11087.8105469, 11098.3027344
3: -5660.2500000, 3688.5415039, -5998.0063477, 3874.5974121, -9534.8476562, 9686.5478516
4: -4258.2089844, 4757.7622070, -4524.7656250, 4979.8222656, -9238.0312500, 9282.5273438

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347155, upper bound: 8119.1347665
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347153, upper bound: 8119.1347147
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.61 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345237, upper bound: 8119.1341401
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345371, upper bound: 8119.1345868
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1330131, upper bound: 8119.1329789
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1330818, upper bound: 8119.1339025
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345202, upper bound: 8119.1340293
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345339, upper bound: 8119.1345046
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1330270, upper bound: 8119.1328248
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1330819, upper bound: 8119.1337999
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345652, upper bound: 8119.1350564
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345654, upper bound: 8119.1349614
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1347155, upper bound: 8119.1350593
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1347151, upper bound: 8119.1349860
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345652, upper bound: 8119.1347662
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1345654, upper bound: 8119.1347141
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1347155, upper bound: 8119.1347665
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -8119.1347153, upper bound: 8119.1347147

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -23989.3398438, 26332.7792969, -23930.6171875, 26287.9824219, -50277.3203125, 50263.3984375
1: -2634.3862305, 1838.3588867, -2629.2331543, 1834.5972900, -4468.9833984, 4467.5917969
2: -4192.6132812, 4900.4150391, -4182.0297852, 4892.2553711, -9084.8691406, 9082.4453125
3: -4790.3281250, 3097.7773438, -4777.6240234, 3091.8854980, -7882.2138672, 7875.4013672
4: -3614.7060547, 3979.2392578, -3605.0825195, 3972.3813477, -7587.0859375, 7584.3217773

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338971, upper bound: 8119.1325441
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338778, upper bound: 8119.1324521
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -24277.9355469, 26630.3867188, -26937.5917969, 29827.8242188, -54105.7578125, 53567.9765625
1: -2664.5925293, 1860.1212158, -2975.4589844, 2057.9079590, -4722.5004883, 4835.5800781
2: -4242.7500000, 4955.8974609, -4703.8466797, 5550.4116211, -9793.1621094, 9659.7441406
3: -4847.6577148, 3133.5139160, -5369.4379883, 3501.2175293, -8348.8750000, 8502.9511719
4: -3658.3564453, 4024.1831055, -4042.6835938, 4510.1542969, -8168.5097656, 8066.8657227

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339686, upper bound: 8119.1335796
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339322, upper bound: 8119.1335094
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -23806.2929688, 26056.9453125, -24325.3828125, 26700.7675781, -50507.0546875, 50382.3164062
1: -2607.7441406, 1825.1110840, -2670.1435547, 1864.6423340, -4472.3867188, 4495.2543945
2: -4158.3842773, 4855.6337891, -4251.7407227, 4971.7792969, -9130.1601562, 9107.3740234
3: -4753.0859375, 3069.4064941, -4859.5678711, 3141.7419434, -7894.8281250, 7928.9746094
4: -3587.5881348, 3940.4450684, -3666.5769043, 4036.2832031, -7623.8710938, 7607.0219727

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329342, upper bound: 8119.1323760
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329943, upper bound: 8119.1324220
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -24083.8789062, 26344.1699219, -27291.2812500, 30173.3710938, -54257.2500000, 53635.4531250
1: -2636.9562988, 1846.0753174, -3010.5292969, 2085.1496582, -4722.1054688, 4856.6044922
2: -4206.6582031, 4909.2211914, -4766.5566406, 5617.6958008, -9824.3535156, 9675.7773438
3: -4808.3476562, 3103.8085938, -5443.8964844, 3543.5571289, -8351.9033203, 8547.7050781
4: -3629.5974121, 3983.8127441, -4099.0458984, 4563.4726562, -8193.0683594, 8082.8579102

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330820, upper bound: 8119.1334954
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330788, upper bound: 8119.1334981
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -23926.9628906, 26249.0234375, -28694.0566406, 31621.5058594, -55548.4687500, 54943.0781250
1: -2626.4121094, 1833.3913574, -3159.7287598, 2198.3125000, -4824.7246094, 4993.1201172
2: -4181.7451172, 4885.8232422, -5018.5346680, 5885.9663086, -10067.7109375, 9904.3574219
3: -4778.8315430, 3088.7270508, -5731.9873047, 3714.2609863, -8493.0927734, 8820.7148438
4: -3606.1081543, 3966.7971191, -4322.3828125, 4778.4326172, -8384.5400391, 8289.1796875

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345128, upper bound: 8119.1337580
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343292, upper bound: 8119.1337598
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -24207.9082031, 26537.7910156, -31699.8808594, 35132.8046875, -59340.7109375, 58237.6718750
1: -2655.7392578, 1854.5963135, -3504.6726074, 2422.5114746, -5078.2509766, 5359.2690430
2: -4230.5605469, 4939.6708984, -5540.4204102, 6540.4648438, -10771.0253906, 10480.0917969
3: -4834.6689453, 3123.4433594, -6326.5610352, 4121.0781250, -8955.7460938, 9450.0039062
4: -3648.6655273, 4010.4233398, -4762.4980469, 5311.3911133, -8960.0566406, 8772.9218750

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1345257, upper bound: 8119.1343231
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343448, upper bound: 8119.1343234
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -23783.8281250, 26028.9062500, -28443.0312500, 31241.8183594, -55025.6406250, 54471.9375000
1: -2605.0642090, 1823.2460938, -3124.7468262, 2179.4719238, -4784.5346680, 4947.9931641
2: -4154.7128906, 4850.7690430, -4972.4375000, 5820.9726562, -9975.6855469, 9823.2060547
3: -4749.5786133, 3066.4328613, -5683.8315430, 3674.7939453, -8424.3720703, 8750.2646484
4: -3584.7048340, 3936.3654785, -4287.3916016, 4723.3076172, -8308.0126953, 8223.7568359

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330275, upper bound: 8119.1318980
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329378, upper bound: 8119.1319895
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329950, upper bound: 8119.1319890
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -24056.8710938, 26310.7187500, -31438.7929688, 34741.8242188, -58798.6914062, 57749.5117188
1: -2633.7434082, 1843.8476562, -3468.5637207, 2402.8991699, -5036.6425781, 5312.4111328
2: -4202.2431641, 4903.4125977, -5492.8823242, 6472.9907227, -10675.2343750, 10396.2949219
3: -4804.0625000, 3100.2055664, -6276.9985352, 4079.9047852, -8883.9667969, 9377.2041016
4: -3626.1313477, 3978.9733887, -4726.2539062, 5254.4365234, -8880.5664062, 8705.2265625

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330818, upper bound: 8119.1331131
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330817, upper bound: 8119.1331837
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330784, upper bound: 8119.1331824
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -24447.0058594, 26864.0761719, -24881.4375000, 27259.8574219, -51706.8632812, 51745.5156250
1: -2686.7282715, 1874.1739502, -2728.9604492, 1907.1275635, -4593.8559570, 4603.1337891
2: -4274.7368164, 5003.8574219, -4348.4052734, 5079.4062500, -9354.1425781, 9352.2617188
3: -4891.9560547, 3159.5288086, -4976.5844727, 3210.0559082, -8102.0117188, 8136.1132812
4: -3688.0566406, 4063.1245117, -3753.5339355, 4123.2001953, -7811.2568359, 7816.6582031

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340457, upper bound: 8119.1347318
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330921, upper bound: 8119.1343488
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -25165.6738281, 27737.8535156, -25628.6093750, 28245.2402344, -53410.9140625, 53366.4609375
1: -2771.2124023, 1929.3486328, -2822.0192871, 1964.3194580, -4735.5317383, 4751.3681641
2: -4403.4521484, 5162.3608398, -4483.7231445, 5255.5922852, -9659.0439453, 9646.0839844
3: -5037.2548828, 3258.1828613, -5128.1494141, 3317.6254883, -8354.8779297, 8386.3320312
4: -3797.0556641, 4193.4350586, -3865.7490234, 4269.3676758, -8066.4233398, 8059.1840820

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1336617, upper bound: 8119.1338906
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330923, upper bound: 8119.1338733
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -27464.1171875, 30406.2851562, -25166.6953125, 27554.8671875, -55018.9843750, 55572.9804688
1: -3033.6018066, 2098.2448730, -2758.9240723, 1928.6572266, -4962.2587891, 4857.1689453
2: -4797.4423828, 5663.0693359, -4398.1445312, 5134.3164062, -9931.7587891, 10061.2109375
3: -5484.5346680, 3569.8327637, -5033.4580078, 3245.3923340, -8729.9248047, 8603.2900391
4: -4126.0708008, 4601.8349609, -3796.7812500, 4167.6650391, -8293.7363281, 8398.6162109

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1344029, upper bound: 8119.1347332
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1337307, upper bound: 8119.1343475
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -28143.1601562, 31228.3574219, -25901.6191406, 28529.7734375, -56672.9296875, 57129.9765625
1: -3113.2590332, 2150.6020508, -2850.9252930, 1984.9871826, -5098.2460938, 5001.5273438
2: -4919.3457031, 5811.7011719, -4531.3100586, 5308.6943359, -10228.0400391, 10343.0117188
3: -5622.7319336, 3662.5703125, -5182.7343750, 3351.6711426, -8974.4033203, 8845.3046875
4: -4229.9985352, 4723.4663086, -3907.1206055, 4312.3071289, -8542.3056641, 8630.5869141

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1341883, upper bound: 8119.1338909
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1337306, upper bound: 8119.1338767
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -24452.1191406, 26871.5273438, -28450.3125000, 31154.6386719, -55606.7578125, 55321.8398438
1: -2687.4003906, 1874.4862061, -3118.5666504, 2181.1123047, -4868.5122070, 4993.0527344
2: -4276.0703125, 5005.5312500, -4972.7968750, 5807.3090820, -10083.3789062, 9978.3281250
3: -4894.3110352, 3160.4760742, -5687.6108398, 3669.2585449, -8563.5693359, 8848.0869141
4: -3689.4897461, 4064.3884277, -4291.8076172, 4711.6357422, -8401.1230469, 8356.1962891

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340462, upper bound: 8119.1343447
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330923, upper bound: 8119.1341227
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -25156.0234375, 27728.0644531, -29505.0117188, 32475.4550781, -57631.4765625, 57233.0742188
1: -2770.3046875, 1928.5379639, -3245.2509766, 2262.2634277, -5032.5664062, 5173.7885742
2: -4402.2353516, 5160.9467773, -5162.6376953, 6043.8061523, -10446.0410156, 10323.5839844
3: -5036.7207031, 3257.1867676, -5903.5517578, 3816.5300293, -8853.2509766, 9160.7382812
4: -3796.3208008, 4192.2890625, -4453.1323242, 4906.3662109, -8702.6875000, 8645.4208984

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1336627, upper bound: 8119.1337294
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330921, upper bound: 8119.1337297
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -27458.4687500, 30400.5976562, -28786.5488281, 31501.8320312, -58960.3007812, 59187.1484375
1: -3032.9826660, 2097.7397461, -3153.9416504, 2206.4833984, -5239.4653320, 5251.6811523
2: -4797.0468750, 5662.1850586, -5031.7055664, 5872.3002930, -10669.3466797, 10693.8896484
3: -5484.9790039, 3569.1721191, -5755.4208984, 3710.8581543, -9195.8369141, 9324.5927734
4: -4126.1665039, 4601.0683594, -4343.2827148, 4764.2729492, -8890.4394531, 8944.3486328

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1344031, upper bound: 8119.1343456
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1337308, upper bound: 8119.1341208
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -28124.6191406, 31208.7949219, -29830.7949219, 32811.4218750, -60936.0312500, 61039.5898438
1: -3111.2927246, 2149.0959473, -3279.4975586, 2286.8962402, -5398.1889648, 5428.5917969
2: -4916.5913086, 5808.4951172, -5219.6142578, 6106.5903320, -11023.1816406, 11028.1093750
3: -5620.4082031, 3660.3579102, -5969.3496094, 3856.8144531, -9477.2197266, 9629.7070312
4: -4227.9291992, 4720.8862305, -4503.0502930, 4957.1904297, -9185.1191406, 9223.9365234

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1341870, upper bound: 8119.1337307
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1337307, upper bound: 8119.1337298
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.26 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1338971, upper bound: 8119.1325441
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1338778, upper bound: 8119.1324521
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1339686, upper bound: 8119.1335796
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1339322, upper bound: 8119.1335094
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1329342, upper bound: 8119.1323760
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1329943, upper bound: 8119.1324220
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330820, upper bound: 8119.1334954
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330788, upper bound: 8119.1334981
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1345128, upper bound: 8119.1337580
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1343292, upper bound: 8119.1337598
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1345257, upper bound: 8119.1343231
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1343448, upper bound: 8119.1343234
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1329378, upper bound: 8119.1319895
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1329950, upper bound: 8119.1319890
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330817, upper bound: 8119.1331837
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330784, upper bound: 8119.1331824
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1340457, upper bound: 8119.1347318
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330921, upper bound: 8119.1343488
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1336617, upper bound: 8119.1338906
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330923, upper bound: 8119.1338733
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1344029, upper bound: 8119.1347332
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1337307, upper bound: 8119.1343475
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1341883, upper bound: 8119.1338909
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1337306, upper bound: 8119.1338767
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1340462, upper bound: 8119.1343447
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330923, upper bound: 8119.1341227
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1336627, upper bound: 8119.1337294
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1330921, upper bound: 8119.1337297
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1344031, upper bound: 8119.1343456
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1337308, upper bound: 8119.1341208
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1341870, upper bound: 8119.1337307
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 3, lower bound: -8119.1337307, upper bound: 8119.1337298

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23343.7695312, 25601.2773438, -23561.1425781, 25857.5664062, -49201.3281250, 49162.4179688
1: -2561.4995117, 1788.9733887, -2586.7866211, 1806.1225586, -4367.6220703, 4375.7592773
2: -4077.2346191, 4767.9370117, -4115.5361328, 4814.9296875, -8892.1630859, 8883.4707031
3: -4657.2778320, 3013.1845703, -4700.6406250, 3042.8083496, -7700.0859375, 7713.8247070
4: -3514.6430664, 3870.4169922, -3546.8610840, 3908.7265625, -7423.3681641, 7417.2778320

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338632, upper bound: 8119.1322186
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338632, upper bound: 8119.1325437
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -24869.7773438, 27293.1679688, -23633.2324219, 25980.2011719, -50849.9765625, 50926.3984375
1: -2729.9960938, 1905.8919678, -2597.7985840, 1811.7207031, -4541.7167969, 4503.6904297
2: -4346.3686523, 5078.3437500, -4130.3510742, 4834.9858398, -9181.3525391, 9208.6943359
3: -4964.6196289, 3211.0827637, -4718.9643555, 3054.8815918, -8019.5009766, 7930.0468750
4: -3746.9929199, 4123.4086914, -3560.3164062, 3926.1582031, -7673.1513672, 7683.7250977

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338502, upper bound: 8119.1321582
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338499, upper bound: 8119.1324530
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23629.8105469, 25895.3593750, -26568.0351562, 29407.1347656, -53036.9414062, 52463.3945312
1: -2591.3801270, 1810.5687256, -2933.3696289, 2029.3350830, -4620.7143555, 4743.9384766
2: -4126.9672852, 4822.8178711, -4636.9594727, 5473.9169922, -9600.8847656, 9459.7763672
3: -4714.2343750, 3048.5437012, -5291.7968750, 3452.6098633, -8166.8442383, 8340.3408203
4: -3558.0375977, 3914.8708496, -3983.6555176, 4447.4692383, -8005.5068359, 7898.5263672

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339378, upper bound: 8119.1334417
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339378, upper bound: 8119.1335795
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -25142.8730469, 27575.7695312, -26633.7929688, 29503.4980469, -54646.3710938, 54209.5585938
1: -2758.7478027, 1926.4984131, -2942.7949219, 2034.6441650, -4793.3916016, 4869.2934570
2: -4394.0048828, 5131.1088867, -4650.8227539, 5491.0869141, -9885.0917969, 9781.9296875
3: -5019.2827148, 3244.9785156, -5309.1059570, 3462.7426758, -8482.0253906, 8554.0839844
4: -3788.5708008, 4166.0849609, -3996.7954102, 4462.0024414, -8250.5732422, 8162.8793945

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339146, upper bound: 8119.1333816
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339142, upper bound: 8119.1335092
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -23419.7167969, 25642.6406250, -24099.8730469, 26455.4238281, -49875.1367188, 49742.5039062
1: -2565.8906250, 1794.9884033, -2645.4909668, 1847.3570557, -4413.2475586, 4440.4785156
2: -4090.8896484, 4778.2470703, -4211.8183594, 4926.3906250, -9017.2783203, 8990.0654297
3: -4674.6728516, 3020.0815430, -4813.1000977, 3112.7233887, -7787.3964844, 7833.1816406
4: -3527.9504395, 3878.1472168, -3631.3515625, 3999.5854492, -7527.5361328, 7509.4990234

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329177, upper bound: 8119.1320609
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329182, upper bound: 8119.1323769
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -24219.5214844, 26440.5664062, -23710.3359375, 26009.7187500, -50229.2382812, 50150.8984375
1: -2648.8403320, 1856.8000488, -2601.5483398, 1817.8153076, -4466.6547852, 4458.3481445
2: -4230.1611328, 4930.8505859, -4142.6914062, 4846.6562500, -9076.8164062, 9073.5419922
3: -4838.7270508, 3118.6218262, -4735.0556641, 3061.8845215, -7900.6113281, 7853.6777344
4: -3653.1171875, 3999.2260742, -3572.7866211, 3933.5483398, -7586.6655273, 7572.0126953

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329887, upper bound: 8119.1321303
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329891, upper bound: 8119.1324219
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -23694.6093750, 25926.4394531, -27068.8554688, 29931.8652344, -53626.4726562, 52995.2890625
1: -2594.7858887, 1815.7242432, -2986.2297363, 2068.0981445, -4662.8837891, 4801.9531250
2: -4138.7963867, 4831.2705078, -4727.1665039, 5573.0410156, -9711.8339844, 9558.4355469
3: -4729.6479492, 3054.1013184, -5398.2573242, 3515.1538086, -8244.8017578, 8452.3564453
4: -3569.7309570, 3921.0485840, -4064.4797363, 4527.3666992, -8097.0976562, 7985.5268555

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330815, upper bound: 8119.1333851
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329184, upper bound: 8119.1334952
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -24495.2363281, 26725.9550781, -26679.0507812, 29482.3691406, -53977.6054688, 53405.0078125
1: -2677.8798828, 1877.6461182, -2942.1486816, 2038.5021973, -4716.3818359, 4819.7944336
2: -4278.0576172, 4984.0239258, -4658.1806641, 5492.8608398, -9770.9179688, 9642.2041016
3: -4893.5551758, 3152.8151855, -5320.2709961, 3463.7788086, -8357.3339844, 8473.0859375
4: -3694.8479004, 4042.3173828, -4005.9389648, 4460.8764648, -8155.7246094, 8048.2558594

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330786, upper bound: 8119.1333810
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330787, upper bound: 8119.1334980
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23255.3613281, 25531.9257812, -28440.8457031, 31346.3574219, -54601.7187500, 53972.7734375
1: -2553.9187012, 1781.9946289, -3132.0007324, 2178.9560547, -4732.8750000, 4913.9951172
2: -4063.6835938, 4751.1987305, -4973.9877930, 5834.5454102, -9898.2285156, 9725.1855469
3: -4641.9526367, 3003.3994141, -5680.3232422, 3681.7360840, -8323.6884766, 8683.7226562
4: -3502.8176270, 3858.0043945, -4283.5239258, 4736.7998047, -8239.6171875, 8141.5283203

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343230, upper bound: 8119.1336641
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343258, upper bound: 8119.1331682
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -26657.7675781, 29482.9921875, -28118.3476562, 31021.4667969, -57679.2304688, 57601.3398438
1: -2947.0246582, 2044.0927734, -3098.8908691, 2154.3972168, -5101.4218750, 5142.9833984
2: -4661.9306641, 5487.0893555, -4918.2519531, 5773.3535156, -10435.2841797, 10405.3408203
3: -5335.6528320, 3460.3459473, -5616.5903320, 3642.2299805, -8977.8828125, 9076.9365234
4: -4020.8017578, 4451.4194336, -4234.7534180, 4687.4365234, -8708.2373047, 8686.1728516

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340371, upper bound: 8119.1335855
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340201, upper bound: 8119.1331684
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23551.2617188, 25837.5097656, -31446.4199219, 34859.9101562, -58411.1718750, 57283.9257812
1: -2584.9558105, 1804.2679443, -3477.1494141, 2403.0090332, -4987.9648438, 5281.4174805
2: -4115.1674805, 4808.1601562, -5495.8691406, 6489.2192383, -10604.3818359, 10304.0283203
3: -4700.8613281, 3040.0541992, -6274.9589844, 4088.7163086, -8789.5761719, 9315.0136719
4: -3547.6474609, 3904.1518555, -4723.5659180, 5269.9887695, -8817.6367188, 8627.7177734

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343824, upper bound: 8119.1342783
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343870, upper bound: 8119.1339617
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26951.4453125, 29785.7089844, -31121.7597656, 34531.1601562, -61482.6054688, 60907.4609375
1: -2977.8842773, 2066.3283691, -3443.6423340, 2378.3105469, -5356.1948242, 5509.9702148
2: -4713.4594727, 5543.6308594, -5439.7758789, 6427.3212891, -11140.7812500, 10983.4062500
3: -5394.8715820, 3496.6596680, -6210.7744141, 4049.0085449, -9443.8798828, 9707.4335938
4: -4065.8698730, 4497.1074219, -4674.5283203, 5220.0546875, -9285.9248047, 9171.6357422

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340797, upper bound: 8119.1342513
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340795, upper bound: 8119.1339611
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -23452.3085938, 25672.6015625, -28070.2031250, 30819.4765625, -54271.7773438, 53742.8046875
1: -2569.0461426, 1797.3620605, -3082.5712891, 2150.9714355, -4720.0166016, 4879.9331055
2: -4096.7573242, 4784.2163086, -4905.5380859, 5744.5551758, -9841.3115234, 9689.7539062
3: -4682.0107422, 3023.9953613, -5606.1489258, 3625.9299316, -8307.9404297, 8630.1425781
4: -3533.4099121, 3882.7502441, -4229.0180664, 4660.8193359, -8194.2294922, 8111.7685547

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329302, upper bound: 8119.1316255
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329304, upper bound: 8119.1319893
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -24159.9062500, 26365.1523438, -28038.5878906, 30810.6152344, -54970.5234375, 54403.7343750
1: -2641.5041504, 1852.0412598, -3081.3762207, 2148.4294434, -4789.9335938, 4933.4174805
2: -4219.6909180, 4917.8916016, -4902.3613281, 5741.1777344, -9960.8671875, 9820.2529297
3: -4827.4453125, 3110.2951660, -5604.1147461, 3623.7268066, -8451.1708984, 8714.4091797
4: -3644.4870605, 3988.4228516, -4226.6816406, 4658.5576172, -8303.0449219, 8215.1035156

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329925, upper bound: 8119.1316256
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1329926, upper bound: 8119.1319891
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -23721.8320312, 25950.3320312, -31059.4824219, 34312.2421875, -58034.0742188, 57009.8125000
1: -2597.3266602, 1817.6918945, -3425.7072754, 2373.9023438, -4971.2290039, 5243.3994141
2: -4143.7290039, 4836.1376953, -5424.9184570, 6395.2001953, -10538.9287109, 10261.0546875
3: -4735.9365234, 3057.3056641, -6198.1728516, 4030.2951660, -8766.2314453, 9255.4785156
4: -3574.4104004, 3924.7670898, -4667.0439453, 5190.7993164, -8765.2099609, 8591.8085938

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330816, upper bound: 8119.1330778
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330818, upper bound: 8119.1331832
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -24431.8671875, 26646.2050781, -31042.9199219, 34318.5898438, -58750.4570312, 57689.1250000
1: -2670.0910645, 1872.5921631, -3426.0668945, 2372.5402832, -5042.6313477, 5298.6591797
2: -4267.0483398, 4970.3291016, -5424.1542969, 6394.8530273, -10661.9013672, 10394.4833984
3: -4881.6782227, 3143.9626465, -6199.0742188, 4029.9443359, -8911.6201172, 9343.0361328
4: -3685.7829590, 4030.9099121, -4666.8398438, 5190.9458008, -8876.7275391, 8697.7490234

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330787, upper bound: 8119.1330780
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330787, upper bound: 8119.1331829
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23788.8535156, 26225.1425781, -24699.5332031, 27069.6757812, -50858.5195312, 50924.6757812
1: -2621.0253906, 1823.5971680, -2709.7900391, 1893.0098877, -4514.0351562, 4533.3872070
2: -4160.6025391, 4883.3925781, -4316.4423828, 5044.1064453, -9204.7089844, 9199.8349609
3: -4762.1093750, 3081.2412109, -4940.3417969, 3187.5766602, -7949.6860352, 8021.5830078
4: -3587.7790527, 3965.7192383, -3725.8540039, 4094.5002441, -7682.2792969, 7691.5732422

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338147, upper bound: 8119.1345500
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339939, upper bound: 8119.1346493
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340264, upper bound: 8119.1345405
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -22476.5507812, 24459.2656250, -23512.9609375, 25598.9316406, -48075.4843750, 47972.2226562
1: -2452.4543457, 1722.8728027, -2567.2968750, 1801.7458496, -4254.2001953, 4290.1699219
2: -3926.2475586, 4564.3862305, -4106.5673828, 4774.8222656, -8701.0693359, 8670.9531250
3: -4496.5556641, 2888.6950684, -4701.6801758, 3022.1154785, -7518.6708984, 7590.3750000
4: -3394.5810547, 3702.2250977, -3549.6457520, 3873.3471680, -7267.9282227, 7251.8710938

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1313265, upper bound: 8119.1335199
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1313262, upper bound: 8119.1329127
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -24459.0410156, 27048.2226562, -25428.9550781, 28038.7968750, -52497.8359375, 52477.1679688
1: -2700.0151367, 1875.0017090, -2800.9077148, 1948.8073730, -4648.8222656, 4675.9091797
2: -4281.0854492, 5031.7573242, -4448.8310547, 5216.1274414, -9497.2128906, 9480.5878906
3: -4898.3886719, 3173.4382324, -5088.0512695, 3292.7495117, -8191.1372070, 8261.4892578
4: -3689.6447754, 4088.3276367, -3835.0595703, 4237.7924805, -7927.4375000, 7923.3872070

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331877, upper bound: 8119.1338285
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1334881, upper bound: 8119.1337827
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1335700, upper bound: 8119.1337891
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -23110.8359375, 25227.8261719, -24253.7949219, 26564.2421875, -49675.0781250, 49481.6210938
1: -2527.2661133, 1771.5891113, -2658.7226562, 1858.8253174, -4386.0913086, 4430.3115234
2: -4039.6149902, 4704.7050781, -4240.4550781, 4949.3193359, -8988.9345703, 8945.1601562
3: -4625.7211914, 2975.4243164, -4853.4248047, 3128.3579102, -7754.0791016, 7828.8491211
4: -3491.4145508, 3817.6398926, -3661.6584473, 4017.8181152, -7509.2324219, 7479.2983398

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1313267, upper bound: 8119.1334663
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1313262, upper bound: 8119.1327167
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -26816.0566406, 29783.3945312, -24982.3066406, 27362.7050781, -54178.7617188, 54765.7031250
1: -2968.9892578, 2048.2429199, -2739.5134277, 1914.3846436, -4883.3740234, 4787.7563477
2: -4684.9667969, 5545.2641602, -4365.6513672, 5098.6718750, -9783.6367188, 9910.9130859
3: -5356.6181641, 3493.1125488, -4996.4965820, 3222.6782227, -8579.2949219, 8489.6093750
4: -4026.8242188, 4506.4091797, -3768.5339355, 4138.7084961, -8165.5327148, 8274.9433594

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1342019, upper bound: 8119.1345483
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343568, upper bound: 8119.1346488
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1343685, upper bound: 8119.1345405
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25493.2226562, 27998.0761719, -23800.7031250, 25895.9316406, -51389.1445312, 51798.7773438
1: -2799.2978516, 1947.3215332, -2597.4931641, 1823.4447021, -4622.7416992, 4544.8144531
2: -4449.8017578, 5223.0410156, -4156.6987305, 4830.2314453, -9280.0332031, 9379.7402344
3: -5090.6386719, 3297.7250977, -4759.1406250, 3057.7050781, -8148.3437500, 8056.8657227
4: -3834.1638184, 4239.3457031, -3593.4016113, 3918.1992188, -7752.3627930, 7832.7465820

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1322390, upper bound: 8119.1335032
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1322389, upper bound: 8119.1329118
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -27437.8730469, 30532.3261719, -25700.8476562, 28322.2968750, -55760.1718750, 56233.1718750
1: -3041.8024902, 2096.4880371, -2829.7075195, 1969.3804932, -5011.1826172, 4926.1953125
2: -4796.7778320, 5680.9140625, -4496.2109375, 5269.0278320, -10065.8056641, 10177.1250000
3: -5484.2792969, 3577.6005859, -5142.4643555, 3326.6694336, -8810.9492188, 8720.0634766
4: -4122.8261719, 4617.7924805, -3876.3166504, 4280.5664062, -8403.3925781, 8494.1083984

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338695, upper bound: 8119.1338292
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1341438, upper bound: 8119.1337948
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1341678, upper bound: 8119.1337945
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -26094.3085938, 28729.8085938, -24528.7832031, 26850.4550781, -52944.7656250, 53258.5898438
1: -2870.2675781, 1993.4331055, -2687.7993164, 1879.6153564, -4749.8828125, 4681.2324219
2: -4557.6567383, 5355.6943359, -4288.4189453, 5002.6596680, -9560.3144531, 9644.1132812
3: -5213.8803711, 3380.3911133, -4908.3491211, 3162.5722656, -8376.4531250, 8288.7402344
4: -3926.3608398, 4348.6972656, -3703.2778320, 4061.0097656, -7987.3706055, 8051.9750977

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1322385, upper bound: 8119.1334461
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1322388, upper bound: 8119.1327160
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23779.4804688, 26215.7128906, -28036.0761719, 30715.9667969, -54495.4453125, 54251.7851562
1: -2620.0280762, 1822.6984863, -3074.2556152, 2149.0710449, -4769.0991211, 4896.9536133
2: -4159.3046875, 4881.8789062, -4901.5673828, 5724.7163086, -9884.0185547, 9783.4462891
3: -4761.5771484, 3080.1127930, -5610.2041016, 3617.4777832, -8379.0546875, 8690.3164062
4: -3586.9885254, 3964.4804688, -4232.3095703, 4644.2524414, -8231.2412109, 8196.7900391

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338421, upper bound: 8119.1343380
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338284, upper bound: 8119.1341476
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -22470.1816406, 24457.9062500, -27712.5253906, 30372.6816406, -52842.8632812, 52170.4296875
1: -2452.0878906, 1722.2159424, -3038.5478516, 2122.3793945, -4574.4672852, 4760.7636719
2: -3925.7258301, 4564.2031250, -4841.5659180, 5660.1601562, -9585.8837891, 9405.7666016
3: -4496.7500000, 2888.3386230, -5531.4106445, 3575.0285645, -8071.7783203, 8419.7490234
4: -3394.3320312, 3702.1633301, -4174.4179688, 4592.0805664, -7986.4125977, 7876.5795898

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1324303, upper bound: 8119.1340994
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1330923, upper bound: 8119.1341191
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -24432.4492188, 27019.4550781, -29205.1738281, 32167.1914062, -56599.6367188, 56224.6289062
1: -2697.1276855, 1872.7994385, -3213.7661133, 2238.8515625, -4935.9794922, 5086.5644531
2: -4276.7480469, 5026.7958984, -5110.1220703, 5985.0048828, -10261.7529297, 10136.9160156
3: -4894.3666992, 3170.1484375, -5842.6372070, 3779.3254395, -8673.6923828, 9012.7851562
4: -3686.2739258, 4084.2419434, -4406.6796875, 4859.2070312, -8545.4804688, 8490.9218750

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331931, upper bound: 8119.1337294
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1336632, upper bound: 8119.1330912
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1336628, upper bound: 8119.1337294
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -23097.3203125, 25215.9433594, -28236.2558594, 30892.6328125, -53989.9531250, 53452.1992188
1: -2526.1313477, 1770.3745117, -3092.3098145, 2164.4921875, -4690.6235352, 4862.6831055
2: -4037.9831543, 4702.7304688, -4938.1093750, 5756.1918945, -9794.1708984, 9640.8388672
3: -4624.7797852, 2974.0048828, -5651.5976562, 3639.1875000, -8263.9648438, 8625.6025391
4: -3490.3098145, 3816.0522461, -4266.3793945, 4669.7514648, -8160.0610352, 8082.4316406

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1313268, upper bound: 8119.1333691
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1313261, upper bound: 8119.1322378
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -26794.3339844, 29758.6191406, -28368.8378906, 31059.2460938, -57853.5781250, 58127.4570312
1: -2966.5820312, 2046.4699707, -3109.2631836, 2174.2333984, -5140.8154297, 5155.7333984
2: -4681.5727539, 5541.1142578, -4959.8862305, 5788.8974609, -10470.4697266, 10501.0000000
3: -5353.7065430, 3490.2849121, -5677.3447266, 3658.6208496, -9012.3271484, 9167.6298828
4: -4024.3232422, 4503.0170898, -4283.3398438, 4696.1933594, -8720.5166016, 8786.3574219

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1342178, upper bound: 8119.1343382
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1342103, upper bound: 8119.1341477
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25475.2539062, 27983.5585938, -28053.6347656, 30724.8789062, -56200.1328125, 56037.1875000
1: -2797.6147461, 1945.8149414, -3074.4350586, 2148.1174316, -4945.7324219, 5020.2500000
2: -4447.2490234, 5220.2915039, -4901.2685547, 5725.9848633, -10173.2324219, 10121.5605469
3: -5088.4711914, 3295.8232422, -5600.1621094, 3617.2099609, -8705.6796875, 8895.9853516
4: -3832.1745605, 4237.2729492, -4226.7392578, 4645.4296875, -8477.6035156, 8464.0117188

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1334070, upper bound: 8119.1340969
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1337308, upper bound: 8119.1341190
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -27400.0468750, 30491.4726562, -29528.7792969, 32501.4492188, -59901.4921875, 60020.2500000
1: -3037.6743164, 2093.4257812, -3247.8073730, 2263.3442383, -5301.0185547, 5341.2329102
2: -4790.5107422, 5673.5620117, -5166.7055664, 6047.4365234, -10837.9472656, 10840.2646484
3: -5478.0986328, 3572.8217773, -5907.9453125, 3819.3840332, -9297.4824219, 9480.7675781
4: -4117.7802734, 4611.8906250, -4456.2343750, 4909.7602539, -9027.5410156, 9068.1250000

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338779, upper bound: 8119.1337301
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1335852, upper bound: 8119.1321684
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1341404, upper bound: 8119.1337297
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1340509, upper bound: 8119.1335619
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -26071.9453125, 28708.5605469, -28564.3105469, 31230.8007812, -57302.7460938, 57272.8710938
1: -2868.1701660, 1991.5633545, -3126.7985840, 2189.2600098, -5057.4301758, 5118.3618164
2: -4554.4584961, 5351.8598633, -4995.4765625, 5819.4111328, -10373.8691406, 10347.3359375
3: -5211.1640625, 3377.8554688, -5717.6518555, 3679.7158203, -8890.8789062, 9095.5078125
4: -3923.9331055, 4345.7128906, -4316.5634766, 4720.9609375, -8644.8945312, 8662.2763672

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1322391, upper bound: 8119.1333510
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1322389, upper bound: 8119.1322371
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.38 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1338632, upper bound: 8119.1322186
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1338632, upper bound: 8119.1325437
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1338502, upper bound: 8119.1321582
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1338499, upper bound: 8119.1324530
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1339378, upper bound: 8119.1334417
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1339378, upper bound: 8119.1335795
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1339146, upper bound: 8119.1333816
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1339142, upper bound: 8119.1335092
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329177, upper bound: 8119.1320609
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329182, upper bound: 8119.1323769
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329887, upper bound: 8119.1321303
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329891, upper bound: 8119.1324219
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330815, upper bound: 8119.1333851
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329184, upper bound: 8119.1334952
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330786, upper bound: 8119.1333810
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330787, upper bound: 8119.1334980
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1343230, upper bound: 8119.1336641
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1343258, upper bound: 8119.1331682
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1340371, upper bound: 8119.1335855
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1340201, upper bound: 8119.1331684
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1343824, upper bound: 8119.1342783
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1343870, upper bound: 8119.1339617
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1340797, upper bound: 8119.1342513
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1340795, upper bound: 8119.1339611
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329302, upper bound: 8119.1316255
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329304, upper bound: 8119.1319893
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329925, upper bound: 8119.1316256
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1329926, upper bound: 8119.1319891
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330816, upper bound: 8119.1330778
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330818, upper bound: 8119.1331832
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330787, upper bound: 8119.1330780
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330787, upper bound: 8119.1331829
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1339939, upper bound: 8119.1346493
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1340264, upper bound: 8119.1345405
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1313265, upper bound: 8119.1335199
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1313262, upper bound: 8119.1329127
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1334881, upper bound: 8119.1337827
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1335700, upper bound: 8119.1337891
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1313267, upper bound: 8119.1334663
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1313262, upper bound: 8119.1327167
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1343568, upper bound: 8119.1346488
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1343685, upper bound: 8119.1345405
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1322390, upper bound: 8119.1335032
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1322389, upper bound: 8119.1329118
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1341438, upper bound: 8119.1337948
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1341678, upper bound: 8119.1337945
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1322385, upper bound: 8119.1334461
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1322388, upper bound: 8119.1327160
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1338421, upper bound: 8119.1343380
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1338284, upper bound: 8119.1341476
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1324303, upper bound: 8119.1340994
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1330923, upper bound: 8119.1341191
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1336632, upper bound: 8119.1330912
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1336628, upper bound: 8119.1337294
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1313268, upper bound: 8119.1333691
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1313261, upper bound: 8119.1322378
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1342178, upper bound: 8119.1343382
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1342103, upper bound: 8119.1341477
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1334070, upper bound: 8119.1340969
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1337308, upper bound: 8119.1341190
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1341404, upper bound: 8119.1337297
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1340509, upper bound: 8119.1335619
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1322391, upper bound: 8119.1333510
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.38
Output dim: 3, lower bound: -8119.1322389, upper bound: 8119.1322371

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -23343.7695312, 25601.2773438, -23275.8554688, 25562.1738281, -48905.9414062, 48877.1289062
1: -2561.4995117, 1788.9733887, -2556.8251953, 1784.2849121, -4345.7841797, 4345.7983398
2: -4077.2346191, 4767.9370117, -4066.2255859, 4758.9931641, -8836.2265625, 8834.1601562
3: -4657.2778320, 3013.1845703, -4643.9003906, 3007.2282715, -7664.5058594, 7657.0844727
4: -3514.6430664, 3870.4169922, -3504.1589355, 3863.7624512, -7378.4042969, 7374.5756836

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331221, upper bound: 8119.1322109
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331223, upper bound: 8119.1322197
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -23343.7695312, 25601.2773438, -24061.5312500, 26453.5722656, -49797.3437500, 49662.8046875
1: -2561.4995117, 1788.9733887, -2645.3227539, 1844.5491943, -4406.0488281, 4434.2954102
2: -4077.2346191, 4767.9370117, -4206.5219727, 4927.0048828, -9004.2373047, 8974.4560547
3: -4657.2778320, 3013.1845703, -4812.0336914, 3110.6411133, -7767.9189453, 7825.2177734
4: -3514.6430664, 3870.4169922, -3627.5983887, 4001.2634277, -7515.9062500, 7498.0151367

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331225, upper bound: 8119.1325324
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1331225, upper bound: 8119.1325421
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -24869.7773438, 27293.1679688, -23435.4726562, 25778.8535156, -50648.6328125, 50728.6367188
1: -2729.9960938, 1905.8919678, -2577.0446777, 1796.5396729, -4526.5356445, 4482.9365234
2: -4346.3686523, 5078.3437500, -4095.7690430, 4796.4101562, -9142.7783203, 9174.1083984
3: -4964.6196289, 3211.0827637, -4678.4296875, 3030.4409180, -7995.0600586, 7889.5126953
4: -3746.9929199, 4123.4086914, -3529.4226074, 3895.3481445, -7642.3408203, 7652.8310547

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338497, upper bound: 8119.1321582
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338501, upper bound: 8119.1321582
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -24869.7773438, 27293.1679688, -23680.2441406, 26011.2500000, -50881.0273438, 50973.4101562
1: -2729.9960938, 1905.8919678, -2601.6962891, 1815.4633789, -4545.4594727, 4507.5878906
2: -4346.3686523, 5078.3437500, -4139.2739258, 4847.9746094, -9194.3417969, 9217.6152344
3: -4964.6196289, 3211.0827637, -4736.4799805, 3060.3569336, -8024.9765625, 7947.5625000
4: -3746.9929199, 4123.4086914, -3571.0341797, 3935.7043457, -7682.6972656, 7694.4423828

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338501, upper bound: 8119.1324523
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1338499, upper bound: 8119.1324523
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -23629.8105469, 25895.3593750, -26283.1035156, 29115.5332031, -52745.3398438, 52178.4609375
1: -2591.3801270, 1810.5687256, -2903.6208496, 2007.6165771, -4598.9956055, 4714.1894531
2: -4126.9672852, 4822.8178711, -4587.7163086, 5418.7143555, -9545.6816406, 9410.5341797
3: -4714.2343750, 3048.5437012, -5235.2280273, 3417.3400879, -8131.5742188, 8283.7714844
4: -3558.0375977, 3914.8708496, -3941.1274414, 4403.0439453, -7961.0815430, 7855.9980469

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339380, upper bound: 8119.1334425
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339376, upper bound: 8119.1334415
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -23629.8105469, 25895.3593750, -27087.0644531, 30015.1328125, -53644.9414062, 52982.4218750
1: -2591.3801270, 1810.5687256, -2993.6560059, 2069.0446777, -4660.4243164, 4804.2246094
2: -4126.9672852, 4822.8178711, -4730.8500977, 5589.2548828, -9716.2226562, 9553.6650391
3: -4714.2343750, 3048.5437012, -5406.5395508, 3522.9604492, -8237.1943359, 8455.0830078
4: -3558.0375977, 3914.8708496, -4066.6826172, 4542.5000000, -8100.5375977, 7981.5537109

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339379, upper bound: 8119.1335790
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339380, upper bound: 8119.1335796
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -25142.8730469, 27575.7695312, -26437.5234375, 29304.6894531, -54447.5625000, 54013.2929688
1: -2758.7478027, 1926.4984131, -2922.2951660, 2019.5355225, -4778.2832031, 4848.7929688
2: -4394.0048828, 5131.1088867, -4616.5361328, 5453.0239258, -9847.0283203, 9747.6445312
3: -5019.2827148, 3244.9785156, -5268.9721680, 3438.6672363, -8457.9492188, 8513.9511719
4: -3788.5708008, 4166.0849609, -3966.2158203, 4431.6191406, -8220.1894531, 8132.3002930

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339142, upper bound: 8119.1333823
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339142, upper bound: 8119.1333822
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -25142.8730469, 27575.7695312, -26695.8632812, 29545.7460938, -54688.6171875, 54271.6328125
1: -2758.7478027, 1926.4984131, -2947.9768066, 2039.4454346, -4798.1933594, 4874.4746094
2: -4394.0048828, 5131.1088867, -4661.7895508, 5506.0947266, -9900.0996094, 9792.8984375
3: -5019.2827148, 3244.9785156, -5329.2553711, 3469.7822266, -8489.0644531, 8574.2343750
4: -3788.5708008, 4166.0849609, -4009.2736816, 4473.3461914, -8261.9169922, 8175.3583984

Time for backsubstitution: 2.83 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.88 + 414.70 = 420.58 seconds
