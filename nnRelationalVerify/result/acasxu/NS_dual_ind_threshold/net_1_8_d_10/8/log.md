## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.60317678965


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7843938, 3.3221977, -3.7843938, 3.3221977, -7.1065907, 7.1065907)
1: (-14.9787064, 12.8845959, -14.9787064, 12.8845959, -27.8633022, 27.8633022)
2: (-7.4894867, 12.0534105, -7.4894867, 12.0534105, -19.5428963, 19.5428963)
3: (-13.1016846, 11.7157326, -13.1016846, 11.7157326, -24.8174152, 24.8174152)
4: (-9.5921707, 12.2164268, -9.5921707, 12.2164268, -21.8085976, 21.8085976)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.10 + 1.91 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.6042070, upper bound: 20.6042070

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041992, upper bound: 20.6041973
time: 0.62 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.44 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 3, lower bound: -20.6041992, upper bound: 20.6041973
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.3461347, 2.9654913, -3.7843938, 3.3221977, -6.6683321, 6.7498837
1: -13.2283525, 11.5012493, -14.9787064, 12.8845959, -26.1129475, 26.4799557
2: -6.6200395, 10.7311440, -7.4894867, 12.0534105, -18.6734505, 18.2206306
3: -11.5760069, 10.4962263, -13.1016846, 11.7157326, -23.2917385, 23.5979099
4: -8.4712791, 10.9263763, -9.5921707, 12.2164268, -20.6877060, 20.5185471

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
time: 0.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
time: 0.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.0484486, 3.5925496, -3.7843938, 3.3221977, -7.3706460, 7.3769422
1: -16.0087757, 13.9060564, -14.9787064, 12.8845959, -28.8933697, 28.8847599
2: -7.9802833, 13.0727425, -7.4894867, 12.0534105, -20.0336933, 20.5622292
3: -13.9787140, 12.6847849, -13.1016846, 11.7157326, -25.6944466, 25.7864685
4: -10.2256079, 13.2608871, -9.5921707, 12.2164268, -22.4420357, 22.8530579

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
time: 0.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
time: 0.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -20.6041913, upper bound: 20.6041913

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.3461347, 2.9654913, -3.3461347, 2.9654913, -6.3116250, 6.3116250
1: -13.2283525, 11.5012493, -13.2283525, 11.5012493, -24.7296028, 24.7296009
2: -6.6200395, 10.7311440, -6.6200395, 10.7311440, -17.3511829, 17.3511829
3: -11.5760069, 10.4962263, -11.5760069, 10.4962263, -22.0722313, 22.0722313
4: -8.4712791, 10.9263763, -8.4712791, 10.9263763, -19.3976555, 19.3976555

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6022452, upper bound: 20.6040527
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6041992, upper bound: 20.6041973
time: 0.87 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.3461347, 2.9654913, -4.0484486, 3.5925496, -6.9386835, 7.0139389
1: -13.2283525, 11.5012493, -16.0087757, 13.9060564, -27.1344070, 27.5100250
2: -6.6200395, 10.7311440, -7.9802833, 13.0727425, -19.6927814, 18.7114277
3: -11.5760069, 10.4962263, -13.9787140, 12.6847849, -24.2607899, 24.4749413
4: -8.4712791, 10.9263763, -10.2256079, 13.2608871, -21.7321663, 21.1519852

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6022452, upper bound: 20.6040527
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6022452, upper bound: 20.6040527
time: 0.76 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.0484486, 3.5925496, -3.3461347, 2.9654913, -7.0139394, 6.9386845
1: -16.0087757, 13.9060564, -13.2283525, 11.5012493, -27.5100231, 27.1344070
2: -7.9802833, 13.0727425, -6.6200395, 10.7311440, -18.7114277, 19.6927814
3: -13.9787140, 12.6847849, -11.5760069, 10.4962263, -24.4749393, 24.2607899
4: -10.2256079, 13.2608871, -8.4712791, 10.9263763, -21.1519852, 21.7321663

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6039871
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6039871
time: 0.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.0484486, 3.5925496, -4.0484486, 3.5925496, -7.6409979, 7.6409979
1: -16.0087757, 13.9060564, -16.0087757, 13.9060564, -29.9148312, 29.9148312
2: -7.9802833, 13.0727425, -7.9802833, 13.0727425, -21.0530262, 21.0530262
3: -13.9787140, 12.6847849, -13.9787140, 12.6847849, -26.6634979, 26.6634979
4: -10.2256079, 13.2608871, -10.2256079, 13.2608871, -23.4864960, 23.4864960

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6039871
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6041913
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.63 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6022452, upper bound: 20.6040527
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6041992, upper bound: 20.6041973
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6022452, upper bound: 20.6040527
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6022452, upper bound: 20.6040527
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6039871
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6039871
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6039871
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 3, lower bound: -20.6018495, upper bound: 20.6041913

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.4816661, 2.2386546, -3.3461347, 2.9654913, -5.4471564, 5.5847893
1: -9.7818518, 8.6575842, -13.2283525, 11.5012493, -21.2831001, 21.8859367
2: -4.9347491, 8.0548239, -6.6200395, 10.7311440, -15.6658936, 14.6748610
3: -8.5770416, 7.9214907, -11.5760069, 10.4962263, -19.0732670, 19.4974957
4: -6.2435703, 8.2708254, -8.4712791, 10.9263763, -17.1699467, 16.7421036

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6021066, upper bound: 20.6021066
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6021066, upper bound: 20.6040606
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.1748228, 2.8172252, -3.3461347, 2.9654913, -6.1403136, 6.1633596
1: -12.5469027, 10.9336882, -13.2283525, 11.5012493, -24.0481529, 24.1620407
2: -6.2786145, 10.1999693, -6.6200395, 10.7311440, -17.0097580, 16.8200092
3: -10.9814510, 9.9884415, -11.5760069, 10.4962263, -21.4776764, 21.5644436
4: -8.0316782, 10.3891039, -8.4712791, 10.9263763, -18.9580536, 18.8603821

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040606, upper bound: 20.6022512
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040606, upper bound: 20.6042052
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.4816661, 2.2386546, -4.0484486, 3.5925496, -6.0742149, 6.2871032
1: -9.7818518, 8.6575842, -16.0087757, 13.9060564, -23.6879082, 24.6663589
2: -4.9347491, 8.0548239, -7.9802833, 13.0727425, -18.0074921, 16.0351067
3: -8.5770416, 7.9214907, -13.9787140, 12.6847849, -21.2618256, 21.9002037
4: -6.2435703, 8.2708254, -10.2256079, 13.2608871, -19.5044575, 18.4964333

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6020410, upper bound: 20.6017109
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6020410, upper bound: 20.6040527
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.1748228, 2.8172252, -4.0484486, 3.5925496, -6.7673712, 6.8656740
1: -12.5469027, 10.9336882, -16.0087757, 13.9060564, -26.4529591, 26.9424629
2: -6.2786145, 10.1999693, -7.9802833, 13.0727425, -19.3513565, 18.1802521
3: -10.9814510, 9.9884415, -13.9787140, 12.6847849, -23.6662369, 23.9671516
4: -8.0316782, 10.3891039, -10.2256079, 13.2608871, -21.2925644, 20.6147118

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039950, upper bound: 20.6018555
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039950, upper bound: 20.6041973
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.1559923, 2.8367624, -3.3461347, 2.9654913, -6.1214824, 6.1828971
1: -12.4439774, 10.9934502, -13.2283525, 11.5012493, -23.9452229, 24.2218018
2: -6.2452865, 10.2560158, -6.6200395, 10.7311440, -16.9764309, 16.8760548
3: -10.8749313, 10.0562792, -11.5760069, 10.4962263, -21.3711548, 21.6322861
4: -7.9366641, 10.4835777, -8.4712791, 10.9263763, -18.8630409, 18.9548569

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6017109, upper bound: 20.6020410
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6017109, upper bound: 20.6039950
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.8707032, 3.4377990, -3.3461347, 2.9654913, -6.8361945, 6.7839336
1: -15.2999287, 13.3078461, -13.2283525, 11.5012493, -26.8011780, 26.5361977
2: -7.6276689, 12.5117207, -6.6200395, 10.7311440, -18.3588123, 19.1317596
3: -13.3622704, 12.1507788, -11.5760069, 10.4962263, -23.8584957, 23.7267799
4: -9.7732592, 12.6973190, -8.4712791, 10.9263763, -20.6996346, 21.1685982

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040527, upper bound: 20.6022452
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040527, upper bound: 20.6041992
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.1559923, 2.8367624, -4.0484486, 3.5925496, -6.7485409, 6.8852110
1: -12.4439774, 10.9934502, -16.0087757, 13.9060564, -26.3500328, 27.0022259
2: -6.2452865, 10.2560158, -7.9802833, 13.0727425, -19.3180294, 18.2362995
3: -10.8749313, 10.0562792, -13.9787140, 12.6847849, -23.5597134, 24.0349922
4: -7.9366641, 10.4835777, -10.2256079, 13.2608871, -21.1975517, 20.7091808

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6016453, upper bound: 20.6016453
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6016453, upper bound: 20.6039871
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.8707032, 3.4377990, -4.0484486, 3.5925496, -7.4632530, 7.4862475
1: -15.2999287, 13.3078461, -16.0087757, 13.9060564, -29.2059822, 29.3166218
2: -7.6276689, 12.5117207, -7.9802833, 13.0727425, -20.7004108, 20.4920044
3: -13.3622704, 12.1507788, -13.9787140, 12.6847849, -26.0470543, 26.1294861
4: -9.7732592, 12.6973190, -10.2256079, 13.2608871, -23.0341454, 22.9229259

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039871, upper bound: 20.6018495
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039871, upper bound: 20.6041913
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.65 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6021066, upper bound: 20.6021066
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6021066, upper bound: 20.6040606
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6040606, upper bound: 20.6022512
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6040606, upper bound: 20.6042052
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6020410, upper bound: 20.6017109
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6020410, upper bound: 20.6040527
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6039950, upper bound: 20.6018555
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6039950, upper bound: 20.6041973
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6017109, upper bound: 20.6020410
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6017109, upper bound: 20.6039950
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6040527, upper bound: 20.6022452
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6040527, upper bound: 20.6041992
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6016453, upper bound: 20.6016453
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6016453, upper bound: 20.6039871
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6039871, upper bound: 20.6018495
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 3, lower bound: -20.6039871, upper bound: 20.6041913

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.4816661, 2.2386546, -3.1748228, 2.8172252, -5.2988911, 5.4134774
1: -9.7818518, 8.6575842, -12.5469027, 10.9336882, -20.7155399, 21.2044868
2: -4.9347491, 8.0548239, -6.2786145, 10.1999693, -15.1347179, 14.3334370
3: -8.5770416, 7.9214907, -10.9814510, 9.9884415, -18.5654831, 18.9029408
4: -6.2435703, 8.2708254, -8.0316782, 10.3891039, -16.6326733, 16.3024998

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6015417, upper bound: 20.5994510
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5982029, upper bound: 20.5988884
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.1748228, 2.8172252, -2.4816661, 2.2386546, -5.4134774, 5.2988911
1: -12.5469027, 10.9336882, -9.7818518, 8.6575842, -21.2044868, 20.7155399
2: -6.2786145, 10.1999693, -4.9347491, 8.0548239, -14.3334379, 15.1347179
3: -10.9814510, 9.9884415, -8.5770416, 7.9214907, -18.9029408, 18.5654831
4: -8.0316782, 10.3891039, -6.2435703, 8.2708254, -16.3025036, 16.6326752

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6004211, upper bound: 20.6018340
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040603, upper bound: 20.6022135
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.1748228, 2.8172252, -3.1748228, 2.8172252, -5.9920478, 5.9920483
1: -12.5469027, 10.9336882, -12.5469027, 10.9336882, -23.4805908, 23.4805908
2: -6.2786145, 10.1999693, -6.2786145, 10.1999693, -16.4785843, 16.4785843
3: -10.9814510, 9.9884415, -10.9814510, 9.9884415, -20.9698906, 20.9698906
4: -8.0316782, 10.3891039, -8.0316782, 10.3891039, -18.4207821, 18.4207802

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6004211, upper bound: 20.6037880
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6040603, upper bound: 20.6041648
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.4816661, 2.2386546, -3.8707032, 3.4377990, -5.9194646, 6.1093578
1: -9.7818518, 8.6575842, -15.2999287, 13.3078461, -23.0896988, 23.9575119
2: -4.9347491, 8.0548239, -7.6276689, 12.5117207, -17.4464703, 15.6824923
3: -8.5770416, 7.9214907, -13.3622704, 12.1507788, -20.7278156, 21.2837601
4: -6.2435703, 8.2708254, -9.7732592, 12.6973190, -18.9408875, 18.0440826

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6015417, upper bound: 20.6026863
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5983688, upper bound: 20.6021238
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.1748228, 2.8172252, -3.1559923, 2.8367624, -6.0115852, 5.9732170
1: -12.5469027, 10.9336882, -12.4439774, 10.9934502, -23.5403519, 23.3776646
2: -6.2786145, 10.1999693, -6.2452865, 10.2560158, -16.5346298, 16.4452553
3: -10.9814510, 9.9884415, -10.8749313, 10.0562792, -21.0377312, 20.8633671
4: -8.0316782, 10.3891039, -7.9366641, 10.4835777, -18.5152493, 18.3257675

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6003557, upper bound: 20.6014383
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039949, upper bound: 20.6018178
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.1748228, 2.8172252, -3.8707032, 3.4377990, -6.6126218, 6.6879282
1: -12.5469027, 10.9336882, -15.2999287, 13.3078461, -25.8547478, 26.2336159
2: -6.2786145, 10.1999693, -7.6276689, 12.5117207, -18.7903347, 17.8276367
3: -10.9814510, 9.9884415, -13.3622704, 12.1507788, -23.1322250, 23.3507099
4: -8.0316782, 10.3891039, -9.7732592, 12.6973190, -20.7289944, 20.1623631

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6003557, upper bound: 20.6037584
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6039949, upper bound: 20.6018178
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.1559923, 2.8367624, -3.1748228, 2.8172252, -5.9732170, 6.0115852
1: -12.4439774, 10.9934502, -12.5469027, 10.9336882, -23.3776665, 23.5403519
2: -6.2452865, 10.2560158, -6.2786145, 10.1999693, -16.4452553, 16.5346298
3: -10.8749313, 10.0562792, -10.9814510, 9.9884415, -20.8633671, 21.0377312
4: -7.9366641, 10.4835777, -8.0316782, 10.3891039, -18.3257675, 18.5152493

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6011460, upper bound: 20.5993877
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5982029, upper bound: 20.5983688
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.8707032, 3.4377990, -2.4816661, 2.2386546, -6.1093578, 5.9194651
1: -15.2999287, 13.3078461, -9.7818518, 8.6575842, -23.9575119, 23.0896988
2: -7.6276689, 12.5117207, -4.9347491, 8.0548239, -15.6824923, 17.4464703
3: -13.3622704, 12.1507788, -8.5770416, 7.9214907, -21.2837601, 20.7278175
4: -9.7732592, 12.6973190, -6.2435703, 8.2708254, -18.0440845, 18.9408894

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034878, upper bound: 20.5989064
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5987192
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.8707032, 3.4377990, -3.1748228, 2.8172252, -6.6879282, 6.6126218
1: -15.2999287, 13.3078461, -12.5469027, 10.9336882, -26.2336159, 25.8547478
2: -7.6276689, 12.5117207, -6.2786145, 10.1999693, -17.8276367, 18.7903347
3: -13.3622704, 12.1507788, -10.9814510, 9.9884415, -23.3507099, 23.1322250
4: -9.7732592, 12.6973190, -8.0316782, 10.3891039, -20.1623631, 20.7289944

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034878, upper bound: 20.5995919
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5994046
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.1559923, 2.8367624, -3.8707032, 3.4377990, -6.5937910, 6.7074656
1: -12.4439774, 10.9934502, -15.2999287, 13.3078461, -25.7518234, 26.2933788
2: -6.2452865, 10.2560158, -7.6276689, 12.5117207, -18.7570076, 17.8836842
3: -10.8749313, 10.0562792, -13.3622704, 12.1507788, -23.0257015, 23.4185486
4: -7.9366641, 10.4835777, -9.7732592, 12.6973190, -20.6339817, 20.2568321

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6011460, upper bound: 20.6018451
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5982029, upper bound: 20.6004314
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.8707032, 3.4377990, -3.1559923, 2.8367624, -6.7074656, 6.5937910
1: -15.2999287, 13.3078461, -12.4439774, 10.9934502, -26.2933788, 25.7518234
2: -7.6276689, 12.5117207, -6.2452865, 10.2560158, -17.8836842, 18.7570076
3: -13.3622704, 12.1507788, -10.8749313, 10.0562792, -23.4185486, 23.0257015
4: -9.7732592, 12.6973190, -7.9366641, 10.4835777, -20.2568321, 20.6339817

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6011460, upper bound: 20.5998225
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6022896, upper bound: 20.5996718
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.8707032, 3.4377990, -3.8707032, 3.4377990, -7.3085022, 7.3085022
1: -15.2999287, 13.3078461, -15.2999287, 13.3078461, -28.6077747, 28.6077747
2: -7.6276689, 12.5117207, -7.6276689, 12.5117207, -20.1393890, 20.1393890
3: -13.3622704, 12.1507788, -13.3622704, 12.1507788, -25.5130444, 25.5130444
4: -9.7732592, 12.6973190, -9.7732592, 12.6973190, -22.4705734, 22.4705772

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035382, upper bound: 20.6026624
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6022896, upper bound: 20.6026380
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.59 seconds
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6015417, upper bound: 20.5994510
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.5982029, upper bound: 20.5988884
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6004211, upper bound: 20.6018340
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6040603, upper bound: 20.6022135
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6004211, upper bound: 20.6037880
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6040603, upper bound: 20.6041648
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6015417, upper bound: 20.6026863
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.5983688, upper bound: 20.6021238
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6003557, upper bound: 20.6014383
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6039949, upper bound: 20.6018178
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6003557, upper bound: 20.6037584
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6039949, upper bound: 20.6018178
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6011460, upper bound: 20.5993877
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.5982029, upper bound: 20.5983688
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6034878, upper bound: 20.5989064
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5987192
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6034878, upper bound: 20.5995919
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5994046
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6011460, upper bound: 20.6018451
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.5982029, upper bound: 20.6004314
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6011460, upper bound: 20.5998225
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6022896, upper bound: 20.5996718
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6035382, upper bound: 20.6026624
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 3, lower bound: -20.6022896, upper bound: 20.6026380

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.0883842, 2.7413933, -2.4816661, 2.2386546, -5.3270388, 5.2230597
1: -12.2035370, 10.6351013, -9.7818518, 8.6575842, -20.8611202, 20.4169540
2: -6.1065154, 9.9293623, -4.9347491, 8.0548239, -14.1613388, 14.8641109
3: -10.6836596, 9.7188129, -8.5770416, 7.9214907, -18.6051502, 18.2958546
4: -7.8119130, 10.1148758, -6.2435703, 8.2708254, -16.0827370, 16.3584461

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5992782, upper bound: 20.6016189
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5987175, upper bound: 20.5982801
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.8787694, 2.5514994, -3.1748228, 2.8172252, -5.6959944, 5.7263222
1: -11.3781652, 9.8942156, -12.5469027, 10.9336882, -22.3118534, 22.4411182
2: -5.7001657, 9.2384615, -6.2786145, 10.1999693, -15.9001331, 15.5170765
3: -9.9726439, 9.0366688, -10.9814510, 9.9884415, -19.9610825, 20.0181179
4: -7.2823725, 9.4121666, -8.0316782, 10.3891039, -17.6714725, 17.4438438

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6001123, upper bound: 20.6001487
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6001123, upper bound: 20.6037880
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.0883842, 2.7413933, -3.1748228, 2.8172252, -5.9056091, 5.9162159
1: -12.2035370, 10.6351013, -12.5469027, 10.9336882, -23.1372261, 23.1820030
2: -6.1065154, 9.9293623, -6.2786145, 10.1999693, -16.3064842, 16.2079735
3: -10.6836596, 9.7188129, -10.9814510, 9.9884415, -20.6721001, 20.7002640
4: -7.8119130, 10.1148758, -8.0316782, 10.3891039, -18.2010174, 18.1465530

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037087, upper bound: 20.6005112
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6037087, upper bound: 20.6041648
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.0883842, 2.7413933, -3.1559923, 2.8367624, -5.9251466, 5.8973856
1: -12.2035370, 10.6351013, -12.4439774, 10.9934502, -23.1969872, 23.0790787
2: -6.1065154, 9.9293623, -6.2452865, 10.2560158, -16.3625317, 16.1746445
3: -10.6836596, 9.7188129, -10.8749313, 10.0562792, -20.7399387, 20.5937443
4: -7.8119130, 10.1148758, -7.9366641, 10.4835777, -18.2954865, 18.0515404

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5992167, upper bound: 20.6012232
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5988833, upper bound: 20.5992328
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.8787694, 2.5514994, -3.8707032, 3.4377990, -6.3165684, 6.4222026
1: -11.3781652, 9.8942156, -15.2999287, 13.3078461, -24.6860123, 25.1941433
2: -5.7001657, 9.2384615, -7.6276689, 12.5117207, -18.2118835, 16.8661270
3: -9.9726439, 9.0366688, -13.3622704, 12.1507788, -22.1234169, 22.3989391
4: -7.2823725, 9.4121666, -9.7732592, 12.6973190, -19.9796867, 19.1854229

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5994904, upper bound: 20.6035067
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5990937, upper bound: 20.5999053
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.0883842, 2.7413933, -3.8707032, 3.4377990, -6.5261831, 6.6120968
1: -12.2035370, 10.6351013, -15.2999287, 13.3078461, -25.5113831, 25.9350300
2: -6.1065154, 9.9293623, -7.6276689, 12.5117207, -18.6182365, 17.5570259
3: -10.6836596, 9.7188129, -13.3622704, 12.1507788, -22.8344364, 23.0810833
4: -7.8119130, 10.1148758, -9.7732592, 12.6973190, -20.5092297, 19.8881340

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5993357, upper bound: 20.6031344
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5989704, upper bound: 20.5999049
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.7385240, 3.3267026, -2.4816661, 2.2386546, -5.9771786, 5.8083687
1: -14.7713270, 12.8762598, -9.7818518, 8.6575842, -23.4289112, 22.6581097
2: -7.3640194, 12.1028013, -4.9347491, 8.0548239, -15.4188433, 17.0375500
3: -12.9016151, 11.7661133, -8.5770416, 7.9214907, -20.8231049, 20.3431511
4: -9.4372253, 12.2898855, -6.2435703, 8.2708254, -17.7080479, 18.5334549

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5987192
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5987192
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.7385240, 3.3267026, -3.1748228, 2.8172252, -6.5557485, 6.5015254
1: -14.7713270, 12.8762598, -12.5469027, 10.9336882, -25.7050152, 25.4231606
2: -7.3640194, 12.1028013, -6.2786145, 10.1999693, -17.5639877, 18.3814163
3: -12.9016151, 11.7661133, -10.9814510, 9.9884415, -22.8900566, 22.7475605
4: -9.4372253, 12.2898855, -8.0316782, 10.3891039, -19.8263283, 20.3215637

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5995008
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993775
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.7385240, 3.3267026, -3.8707032, 3.4377990, -7.1763225, 7.1974058
1: -14.7713270, 12.8762598, -15.2999287, 13.3078461, -28.0791740, 28.1761837
2: -7.3640194, 12.1028013, -7.6276689, 12.5117207, -19.8757401, 19.7304707
3: -12.9016151, 11.7661133, -13.3622704, 12.1507788, -25.0523930, 25.1283817
4: -9.4372253, 12.2898855, -9.7732592, 12.6973190, -22.1345406, 22.0631428

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026400, upper bound: 20.5993215
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026380
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.60 seconds
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5992782, upper bound: 20.6016189
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5987175, upper bound: 20.5982801
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6001123, upper bound: 20.6001487
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6001123, upper bound: 20.6037880
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6037087, upper bound: 20.6005112
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6037087, upper bound: 20.6041648
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5992167, upper bound: 20.6012232
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5988833, upper bound: 20.5992328
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5994904, upper bound: 20.6035067
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5990937, upper bound: 20.5999053
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5993357, upper bound: 20.6031344
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.5989704, upper bound: 20.5999049
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5987192
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6021238, upper bound: 20.5987192
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6035437, upper bound: 20.5995008
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6035433, upper bound: 20.5993775
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6026400, upper bound: 20.5993215
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 3, lower bound: -20.6026400, upper bound: 20.6026380

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.8787694, 2.5514994, -3.0883842, 2.7413933, -5.6201630, 5.6398835
1: -11.3781652, 9.8942156, -12.2035370, 10.6351013, -22.0132675, 22.0977478
2: -5.7001657, 9.2384615, -6.1065154, 9.9293623, -15.6295280, 15.3449764
3: -9.9726439, 9.0366688, -10.6836596, 9.7188129, -19.6914558, 19.7203293
4: -7.2823725, 9.4121666, -7.8119130, 10.1148758, -17.3972454, 17.2240753

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5999337, upper bound: 20.5990098
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5989180, upper bound: 20.5987950
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.0883842, 2.7413933, -2.8787694, 2.5514994, -5.6398835, 5.6201630
1: -12.2035370, 10.6351013, -11.3781652, 9.8942156, -22.0977497, 22.0132675
2: -6.1065154, 9.9293623, -5.7001657, 9.2384615, -15.3449764, 15.6295271
3: -10.6836596, 9.7188129, -9.9726439, 9.0366688, -19.7203293, 19.6914558
4: -7.8119130, 10.1148758, -7.2823725, 9.4121666, -17.2240772, 17.3972454

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035729, upper bound: 20.5993892
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0883842, 2.7413933, -3.0883842, 2.7413933, -5.8297777, 5.8297777
1: -12.2035370, 10.6351013, -12.2035370, 10.6351013, -22.8386364, 22.8386383
2: -6.1065154, 9.9293623, -6.1065154, 9.9293623, -16.0358753, 16.0358772
3: -10.6836596, 9.7188129, -10.6836596, 9.7188129, -20.4024734, 20.4024734
4: -7.8119130, 10.1148758, -7.8119130, 10.1148758, -17.9267883, 17.9267883

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035729, upper bound: 20.5993892
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.8787694, 2.5514994, -3.7385240, 3.3267026, -6.2054720, 6.2900224
1: -11.3781652, 9.8942156, -14.7713270, 12.8762598, -24.2544193, 24.6655407
2: -5.7001657, 9.2384615, -7.3640194, 12.1028013, -17.8029652, 16.6024818
3: -9.9726439, 9.0366688, -12.9016151, 11.7661133, -21.7387524, 21.9382839
4: -7.2823725, 9.4121666, -9.4372253, 12.2898855, -19.5722561, 18.8493881

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5990937, upper bound: 20.5999053
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5990937, upper bound: 20.5999053
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.7385240, 3.3267026, -2.8787694, 2.5514994, -6.2900224, 6.2054720
1: -14.7713270, 12.8762598, -11.3781652, 9.8942156, -24.6655388, 24.2544193
2: -7.3640194, 12.1028013, -5.7001657, 9.2384615, -16.6024818, 17.8029652
3: -12.9016151, 11.7661133, -9.9726439, 9.0366688, -21.9382839, 21.7387524
4: -9.4372253, 12.2898855, -7.2823725, 9.4121666, -18.8493881, 19.5722561

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5992870
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5993775
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.7385240, 3.3267026, -3.0883842, 2.7413933, -6.4799170, 6.4150867
1: -14.7713270, 12.8762598, -12.2035370, 10.6351013, -25.4064293, 25.0797901
2: -7.3640194, 12.1028013, -6.1065154, 9.9293623, -17.2933807, 18.2093163
3: -12.9016151, 11.7661133, -10.6836596, 9.7188129, -22.6204281, 22.4497719
4: -9.4372253, 12.2898855, -7.8119130, 10.1148758, -19.5521011, 20.1017990

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5992870
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5993775
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.65 seconds
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.5999337, upper bound: 20.5990098
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.5989180, upper bound: 20.5987950
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.6035729, upper bound: 20.5993892
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.6035729, upper bound: 20.5993892
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.5990937, upper bound: 20.5999053
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.5990937, upper bound: 20.5999053
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5992870
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5993775
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5992870
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.65
Output dim: 3, lower bound: -20.6027677, upper bound: 20.5993775

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.9774008, 2.6441269, -2.8787694, 2.5514994, -5.5289001, 5.5228963
1: -11.7608309, 10.2618475, -11.3781652, 9.8942156, -21.6550426, 21.6400127
2: -5.8837781, 9.5808554, -5.7001657, 9.2384615, -15.1222401, 15.2810211
3: -10.2994843, 9.3810072, -9.9726439, 9.0366688, -19.3361473, 19.3536491
4: -7.5304580, 9.7631979, -7.2823725, 9.4121666, -16.9426193, 17.0455685

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5914292, upper bound: 20.5987402
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035732, upper bound: 20.5995122
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034011, upper bound: 20.5984377
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.9774008, 2.6441269, -3.0883842, 2.7413933, -5.7187939, 5.7325110
1: -11.7608309, 10.2618475, -12.2035370, 10.6351013, -22.3959312, 22.4653816
2: -5.8837781, 9.5808554, -6.1065154, 9.9293623, -15.8131409, 15.6873703
3: -10.2994843, 9.3810072, -10.6836596, 9.7188129, -20.0182972, 20.0646667
4: -7.5304580, 9.7631979, -7.8119130, 10.1148758, -17.6453304, 17.5751095

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
time: 0.73 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.68 seconds
NS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.68
Output dim: 3, lower bound: -20.6035732, upper bound: 20.5995122
NS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.68
Output dim: 3, lower bound: -20.6034011, upper bound: 20.5984377
NS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.68
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947
NS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.68
Output dim: 3, lower bound: -20.5987947, upper bound: 20.5987947

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.9774008, 2.6441269, -2.5868821, 2.3127601, -5.2901611, 5.2310085
1: -11.7608309, 10.2618475, -10.1806831, 8.9724159, -20.7332439, 20.4425316
2: -5.8837781, 9.5808554, -5.1232977, 8.3946505, -14.2784290, 14.7041531
3: -10.2994843, 9.3810072, -8.9306316, 8.2380486, -18.5375309, 18.3116379
4: -7.5304580, 9.7631979, -6.5389714, 8.5889101, -16.1193676, 16.3021698

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035732, upper bound: 20.5995122
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033286, upper bound: 20.5995108
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.9774008, 2.6441269, -2.8223763, 2.5066087, -5.4840097, 5.4665031
1: -11.7608309, 10.2618475, -11.1499701, 9.7224941, -21.4833221, 21.4118176
2: -5.8837781, 9.5808554, -5.5917454, 9.0794430, -14.9632206, 15.1726007
3: -10.2994843, 9.3810072, -9.7727051, 8.8857441, -19.1852283, 19.1537094
4: -7.5304580, 9.7631979, -7.1298919, 9.2559719, -16.7864304, 16.8930893

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6034011, upper bound: 20.5984377
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6031564, upper bound: 20.5984376
time: 0.65 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.51 seconds
NS_A1_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.51
Output dim: 3, lower bound: -20.6035732, upper bound: 20.5995122
NS_A1_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.51
Output dim: 3, lower bound: -20.6033286, upper bound: 20.5995108
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.51
Output dim: 3, lower bound: -20.6034011, upper bound: 20.5984377
NS_A1_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.51
Output dim: 3, lower bound: -20.6031564, upper bound: 20.5984376

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.7628362, 2.4637318, -2.5868821, 2.3127601, -5.0755963, 5.0506129
1: -10.8943176, 9.5599117, -10.1806831, 8.9724159, -19.8667336, 19.7405949
2: -5.4570904, 8.9419165, -5.1232977, 8.3946505, -13.8517389, 14.0652142
3: -9.5473280, 8.7534590, -8.9306316, 8.2380486, -17.7853775, 17.6840897
4: -6.9823217, 9.1207476, -6.5389714, 8.5889101, -15.5712309, 15.6597166

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035728, upper bound: 20.5994460
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6024840, upper bound: 20.5989408
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6035002, upper bound: 20.5995098
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.9136121, 2.5920463, -2.5868821, 2.3127601, -5.2263718, 5.1789260
1: -11.5117750, 10.0242500, -10.1806831, 8.9724159, -20.4841900, 20.2049332
2: -5.7628946, 9.4281664, -5.1232977, 8.3946505, -14.1575432, 14.5514631
3: -10.0787239, 9.1698151, -8.9306316, 8.2380486, -18.3167706, 18.1004467
4: -7.3596964, 9.5936937, -6.5389714, 8.5889101, -15.9486065, 16.1326637

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029323, upper bound: 20.5994209
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6026335, upper bound: 20.5989806
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032235, upper bound: 20.5995086
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.7628362, 2.4637318, -2.8223763, 2.5066087, -5.2694449, 5.2861080
1: -10.8943176, 9.5599117, -11.1499701, 9.7224941, -20.6168118, 20.7098808
2: -5.4570904, 8.9419165, -5.5917454, 9.0794430, -14.5365314, 14.5336618
3: -9.5473280, 8.7534590, -9.7727051, 8.8857441, -18.4330711, 18.5261650
4: -6.9823217, 9.1207476, -7.1298919, 9.2559719, -16.2382927, 16.2506371

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6033916, upper bound: 20.5983712
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032410, upper bound: 20.5975142
time: 0.84 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 4.73 seconds
NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 3, lower bound: -20.6024840, upper bound: 20.5989408
NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 3, lower bound: -20.6035002, upper bound: 20.5995098
NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 3, lower bound: -20.6026335, upper bound: 20.5989806
NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 3, lower bound: -20.6032235, upper bound: 20.5995086
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 3, lower bound: -20.6033916, upper bound: 20.5983712
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 3, lower bound: -20.6032410, upper bound: 20.5975142

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.7628362, 2.4637318, -2.5600464, 2.2897873, -5.0526237, 5.0237770
1: -10.8943176, 9.5599117, -10.0737877, 8.8818312, -19.7761497, 19.6336975
2: -5.4570904, 8.9419165, -5.0693245, 8.3115854, -13.7686758, 14.0112391
3: -9.5473280, 8.7534590, -8.8382721, 8.1559944, -17.7033215, 17.5917320
4: -6.9823217, 9.1207476, -6.4711933, 8.5056515, -15.4879732, 15.5919399

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6028839, upper bound: 20.5995098
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6028839, upper bound: 20.5995098
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.9136121, 2.5920463, -2.5600464, 2.2897873, -5.2033992, 5.1520905
1: -11.5117750, 10.0242500, -10.0737877, 8.8818312, -20.3936043, 20.0980377
2: -5.7628946, 9.4281664, -5.0693245, 8.3115854, -14.0744772, 14.4974871
3: -10.0787239, 9.1698151, -8.8382721, 8.1559944, -18.2347164, 18.0080853
4: -7.3596964, 9.5936937, -6.4711933, 8.5056515, -15.8653479, 16.0648861

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029904, upper bound: 20.5995086
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.6029904, upper bound: 20.5995086
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.7628362, 2.4637318, -2.7345996, 2.4355817, -5.1984177, 5.1983314
1: -10.8943176, 9.5599117, -10.8078384, 9.4145479, -20.3088646, 20.3677502
2: -5.4570904, 8.9419165, -5.4122515, 8.8252964, -14.2823849, 14.3541679
3: -9.5473280, 8.7534590, -9.4779425, 8.6099148, -18.1572418, 18.2314014
4: -6.9823217, 9.1207476, -6.9046526, 8.9995193, -15.9818411, 16.0254002

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.7628362, 2.4637318, -2.7812736, 2.4718451, -5.2346811, 5.2450051
1: -10.8943176, 9.5599117, -10.9866199, 9.5847225, -20.4790401, 20.5465317
2: -5.4570904, 8.9419165, -5.5103316, 8.9551115, -14.4122009, 14.4522476
3: -9.5473280, 8.7534590, -9.6306286, 8.7631578, -18.3104858, 18.3840866
4: -6.9823217, 9.1207476, -7.0237837, 9.1310711, -16.1133919, 16.1445312

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
time: 0.79 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 3.11 seconds
NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6028839, upper bound: 20.5995098
NS_A1_B1_A2_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6028839, upper bound: 20.5995098
NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6029904, upper bound: 20.5995086
NS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6029904, upper bound: 20.5995086
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142
NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.11
Output dim: 3, lower bound: -20.6032382, upper bound: 20.5975142

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.6952832, 2.4085646, -2.7345996, 2.4355817, -5.1308651, 5.1431642
1: -10.6350641, 9.3137541, -10.8078384, 9.4145479, -20.0496120, 20.1215935
2: -5.3205199, 8.7457218, -5.4122515, 8.8252964, -14.1458149, 14.1579723
3: -9.3250933, 8.5306492, -9.4779425, 8.6099148, -17.9350071, 18.0085907
4: -6.8029838, 8.9134340, -6.9046526, 8.9995193, -15.8025036, 15.8180866

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.7182889, 2.4263511, -2.7345996, 2.4355817, -5.1538706, 5.1609507
1: -10.7172976, 9.4103794, -10.8078384, 9.4145479, -20.1318455, 20.2182159
2: -5.3689971, 8.8077259, -5.4122515, 8.8252964, -14.1942940, 14.2199774
3: -9.3935499, 8.6200275, -9.4779425, 8.6099148, -18.0034637, 18.0979691
4: -6.8677702, 8.9853649, -6.9046526, 8.9995193, -15.8672895, 15.8900166

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.6952832, 2.4085646, -2.7812736, 2.4718451, -5.1671286, 5.1898379
1: -10.6350641, 9.3137541, -10.9866199, 9.5847225, -20.2197857, 20.3003731
2: -5.3205199, 8.7457218, -5.5103316, 8.9551115, -14.2756310, 14.2560530
3: -9.3250933, 8.5306492, -9.6306286, 8.7631578, -18.0882492, 18.1612778
4: -6.8029838, 8.9134340, -7.0237837, 9.1310711, -15.9340544, 15.9372177

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.7182889, 2.4263511, -2.7812736, 2.4718451, -5.1901340, 5.2076240
1: -10.7172976, 9.4103794, -10.9866199, 9.5847225, -20.3020210, 20.3969975
2: -5.3689971, 8.8077259, -5.5103316, 8.9551115, -14.3241081, 14.3180580
3: -9.3935499, 8.6200275, -9.6306286, 8.7631578, -18.1567078, 18.2506561
4: -6.8677702, 8.9853649, -7.0237837, 9.1310711, -15.9988413, 16.0091476

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.01 + 159.42 = 162.42 seconds
