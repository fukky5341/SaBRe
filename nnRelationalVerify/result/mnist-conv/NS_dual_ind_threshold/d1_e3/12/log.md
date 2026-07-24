## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.07702695


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7833967, -6.1830349, -6.7833967, -6.1830349, -0.2667882, 0.2667881)
1: (-6.3563976, -5.9922609, -6.3563976, -5.9922609, -0.2112004, 0.2112004)
2: (-3.6290519, -3.2717590, -3.6290519, -3.2717590, -0.1196281, 0.1196281)
3: (-4.2911897, -3.9070742, -4.2911897, -3.9070742, -0.1597021, 0.1597021)
4: (-7.7611260, -7.3932252, -7.7611260, -7.3932252, -0.2205417, 0.2205417)
5: (-9.8297024, -9.3951015, -9.8297024, -9.3951015, -0.1746140, 0.1746137)
6: (-12.6773815, -12.1147928, -12.6773815, -12.1147928, -0.2252733, 0.2252733)
7: (3.7210720, 4.0095162, 3.7210720, 4.0095162, -0.2558870, 0.2558873)
8: (-1.7929845, -1.4484472, -1.7929845, -1.4484472, -0.1682277, 0.1682276)
9: (-1.5299881, -1.2165775, -1.5299881, -1.2165775, -0.2075396, 0.2075393)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.29 + 32.92 = 55.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0778050, upper bound: 0.0778050

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2138
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1759
type: A, layer: 3, pos: 948
type: A, layer: 3, pos: 1976
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2802
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 436
type: A, layer: 3, pos: 1783
type: A, layer: 3, pos: 2936

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2138

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0765702, upper bound: 0.0770706
time: 3.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770705, upper bound: 0.0770706
time: 3.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.55
Output dim: 7, lower bound: -0.0765702, upper bound: 0.0770706
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.55
Output dim: 7, lower bound: -0.0770705, upper bound: 0.0770706

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.7833691, -6.1928616, -6.7833872, -6.1864223, -0.2599835, 0.2475590
1: -6.3563604, -5.9926753, -6.3563838, -5.9924040, -0.2106991, 0.2097651
2: -3.6290410, -3.2738533, -3.6290474, -3.2724819, -0.1177584, 0.1151252
3: -4.2911897, -3.9096823, -4.2911897, -3.9079900, -0.1592753, 0.1588628
4: -7.7578306, -7.3932252, -7.7599912, -7.3932252, -0.2198260, 0.2201473
5: -9.8282709, -9.3951149, -9.8292065, -9.3951073, -0.1709080, 0.1732392
6: -12.6714344, -12.1148071, -12.6753330, -12.1147957, -0.2201375, 0.2232122
7: 3.7232747, 4.0095148, 3.7218313, 4.0095167, -0.2553732, 0.2554588
8: -1.7929831, -1.4506397, -1.7929845, -1.4492044, -0.1673490, 0.1664665
9: -1.5279517, -1.2165775, -1.5292499, -1.2165775, -0.2069392, 0.2072442

Time for backsubstitution: 7.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2138
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1759
type: B, layer: 3, pos: 948
type: B, layer: 3, pos: 1976
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2802
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 436
type: B, layer: 3, pos: 1783
type: B, layer: 3, pos: 2936

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2138

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0765702, upper bound: 0.0765702
time: 3.18 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0765702, upper bound: 0.0770706
time: 3.05 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.7911124, -6.2026405, -6.7833939, -6.1918402, -0.3014417, 0.2464263
1: -6.3559933, -5.9942489, -6.3563933, -5.9931540, -0.2134957, 0.2093171
2: -3.6292255, -3.2799559, -3.6290512, -3.2755709, -0.1306171, 0.1140021
3: -4.2930703, -3.9130688, -4.2911897, -3.9098299, -0.1628208, 0.1594877
4: -7.7577615, -7.3890233, -7.7594585, -7.3932252, -0.2213602, 0.2174273
5: -9.8244438, -9.3953247, -9.8272152, -9.3951035, -0.1700439, 0.1819563
6: -12.6755161, -12.1056099, -12.6765442, -12.1147957, -0.2226487, 0.2365180
7: 3.7240055, 4.0127792, 3.7224772, 4.0095167, -0.2559717, 0.2564583
8: -1.7931333, -1.4553189, -1.7929845, -1.4515386, -0.1703440, 0.1658888
9: -1.5247107, -1.2156515, -1.5276096, -1.2165775, -0.2075129, 0.2050283

Time for backsubstitution: 7.75 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2138
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1759
type: B, layer: 3, pos: 948
type: B, layer: 3, pos: 1976
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2802
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 436
type: B, layer: 3, pos: 1783
type: B, layer: 3, pos: 2936

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 2138

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770705, upper bound: 0.0765702
time: 3.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0770705, upper bound: 0.0770705
time: 3.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.47 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 14.47
Output dim: 7, lower bound: -0.0765702, upper bound: 0.0765702
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.47
Output dim: 7, lower bound: -0.0765702, upper bound: 0.0770706
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.47
Output dim: 7, lower bound: -0.0770705, upper bound: 0.0765702
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.47
Output dim: 7, lower bound: -0.0770705, upper bound: 0.0770705

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.7833691, -6.1928616, -6.7911124, -6.2026405, -0.2656240, 0.2827725
1: -6.3563604, -5.9926753, -6.3559933, -5.9942489, -0.2110515, 0.2121271
2: -3.6290410, -3.2738533, -3.6292255, -3.2799559, -0.1187839, 0.1266498
3: -4.2911897, -3.9096823, -4.2930703, -3.9130688, -0.1567155, 0.1636997
4: -7.7578306, -7.3932252, -7.7577615, -7.3890233, -0.2194968, 0.2151748
5: -9.8282709, -9.3951149, -9.8244438, -9.3953247, -0.1785119, 0.1741185
6: -12.6714344, -12.1148071, -12.6755161, -12.1056099, -0.2320464, 0.2242112
7: 3.7232747, 4.0095148, 3.7240055, 4.0127792, -0.2577922, 0.2524565
8: -1.7929831, -1.4506397, -1.7931333, -1.4553189, -0.1644043, 0.1706816
9: -1.5279517, -1.2165775, -1.5247107, -1.2156515, -0.2080011, 0.2016596

Time for backsubstitution: 7.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1759
type: A, layer: 3, pos: 948
type: A, layer: 3, pos: 1976
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2802
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 436
type: A, layer: 3, pos: 1783
type: A, layer: 3, pos: 2936

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 654

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0746000, upper bound: 0.0769265
time: 3.45 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0760349, upper bound: 0.0765352
time: 3.01 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.7911124, -6.2026405, -6.7833691, -6.1928616, -0.2827725, 0.2656240
1: -6.3559933, -5.9942489, -6.3563604, -5.9926753, -0.2121271, 0.2110515
2: -3.6292255, -3.2799559, -3.6290410, -3.2738533, -0.1266498, 0.1187839
3: -4.2930703, -3.9130688, -4.2911897, -3.9096823, -0.1636997, 0.1567154
4: -7.7577615, -7.3890233, -7.7578306, -7.3932252, -0.2151749, 0.2194971
5: -9.8244438, -9.3953247, -9.8282709, -9.3951149, -0.1741186, 0.1785119
6: -12.6755161, -12.1056099, -12.6714344, -12.1148071, -0.2242113, 0.2320464
7: 3.7240055, 4.0127792, 3.7232747, 4.0095148, -0.2524567, 0.2577922
8: -1.7931333, -1.4553189, -1.7929831, -1.4506397, -0.1706816, 0.1644043
9: -1.5247107, -1.2156515, -1.5279517, -1.2165775, -0.2016594, 0.2080013

Time for backsubstitution: 8.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1759
type: A, layer: 3, pos: 948
type: A, layer: 3, pos: 1976
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2802
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 436
type: A, layer: 3, pos: 1783
type: A, layer: 3, pos: 2936

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 654

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0750951, upper bound: 0.0764262
time: 3.41 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0765352, upper bound: 0.0760348
time: 3.19 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.7911124, -6.2026405, -6.7911124, -6.2026405, -0.2465664, 0.2465665
1: -6.3559933, -5.9942489, -6.3559933, -5.9942489, -0.2094676, 0.2094676
2: -3.6292255, -3.2799559, -3.6292255, -3.2799559, -0.1140300, 0.1140300
3: -4.2930703, -3.9130688, -4.2930703, -3.9130688, -0.1594718, 0.1594716
4: -7.7577615, -7.3890233, -7.7577615, -7.3890233, -0.2213464, 0.2213461
5: -9.8244438, -9.3953247, -9.8244438, -9.3953247, -0.1701152, 0.1701152
6: -12.6755161, -12.1056099, -12.6755161, -12.1056099, -0.2227336, 0.2227335
7: 3.7240055, 4.0127792, 3.7240055, 4.0127792, -0.2559214, 0.2559214
8: -1.7931333, -1.4553189, -1.7931333, -1.4553189, -0.1658705, 0.1658705
9: -1.5247107, -1.2156515, -1.5247107, -1.2156515, -0.2075074, 0.2075074

Time for backsubstitution: 7.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1759
type: A, layer: 3, pos: 948
type: A, layer: 3, pos: 1976
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2802
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 436
type: A, layer: 3, pos: 1783
type: A, layer: 3, pos: 2936

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 654

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0750951, upper bound: 0.0764262
time: 3.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0765353, upper bound: 0.0760349
time: 3.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.40 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 14.40
Output dim: 7, lower bound: -0.0746000, upper bound: 0.0769265
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 14.40
Output dim: 7, lower bound: -0.0760349, upper bound: 0.0765352
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 14.40
Output dim: 7, lower bound: -0.0750951, upper bound: 0.0764262
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 14.40
Output dim: 7, lower bound: -0.0765352, upper bound: 0.0760348
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 14.40
Output dim: 7, lower bound: -0.0750951, upper bound: 0.0764262
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 14.40
Output dim: 7, lower bound: -0.0765353, upper bound: 0.0760349

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 55.21 + 79.01 = 134.22 seconds
