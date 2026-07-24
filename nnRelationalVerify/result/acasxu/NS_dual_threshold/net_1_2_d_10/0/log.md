## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 1757.072574941339


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543)
1: (-485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883)
2: (-554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562)
3: (-788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867)
4: (-931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.51 + 2.05 = 3.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1252887
time: 0.82 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.73 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1252887
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -376.9322205, 1529.5915527, -352.9316101, 1436.0976562, -1813.0297852, 1882.5230713
1: -462.0413818, 1708.4271240, -432.6250916, 1604.0590820, -2066.1000977, 2141.0522461
2: -528.3616943, 1736.7906494, -494.3493042, 1629.8876953, -2158.2495117, 2231.1396484
3: -750.2429199, 1891.5660400, -702.1618652, 1774.0732422, -2524.3161621, 2593.7277832
4: -886.9804077, 1766.1915283, -828.9537354, 1656.8021240, -2543.7819824, 2595.1452637

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1251902
time: 0.67 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1252887
time: 0.69 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -393.8564453, 1599.1993408, -392.9946289, 1595.7901611, -1989.6463623, 1992.1939697
1: -482.7509155, 1786.1520996, -481.7033691, 1782.3461914, -2265.0964355, 2267.8554688
2: -552.3273926, 1815.8070068, -551.1360474, 1811.9309082, -2364.2573242, 2366.9431152
3: -784.2661743, 1977.9160156, -782.5770874, 1973.6967773, -2757.9628906, 2760.4931641
4: -927.2888794, 1846.3841553, -925.2904663, 1842.4506836, -2769.7395020, 2771.6745605

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1251902
time: 0.78 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.07 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1251902
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1252887
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1251902
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -352.9316101, 1436.0976562, -352.9316101, 1436.0976562, -1789.0291748, 1789.0291748
1: -432.6250916, 1604.0590820, -432.6250916, 1604.0590820, -2036.6842041, 2036.6842041
2: -494.3493042, 1629.8876953, -494.3493042, 1629.8876953, -2124.2368164, 2124.2368164
3: -702.1618652, 1774.0732422, -702.1618652, 1774.0732422, -2476.2351074, 2476.2351074
4: -828.9537354, 1656.8021240, -828.9537354, 1656.8021240, -2485.7553711, 2485.7553711

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0677288, upper bound: 1757.0373133
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.0354614
time: 0.62 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -392.9946289, 1595.7901611, -352.9316101, 1436.0976562, -1829.0922852, 1948.7218018
1: -481.7033691, 1782.3461914, -432.6250916, 1604.0590820, -2085.7622070, 2214.9709473
2: -551.1360474, 1811.9309082, -494.3493042, 1629.8876953, -2181.0236816, 2306.2795410
3: -782.5770874, 1973.6967773, -702.1618652, 1774.0732422, -2556.6503906, 2675.8586426
4: -925.2904663, 1842.4506836, -828.9537354, 1656.8021240, -2582.0925293, 2671.4042969

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0677288, upper bound: 1757.0373171
time: 0.64 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.0372694
time: 0.80 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -352.9316101, 1436.0976562, -392.9946289, 1595.7901611, -1948.7218018, 1829.0922852
1: -432.6250916, 1604.0590820, -481.7033691, 1782.3461914, -2214.9707031, 2085.7624512
2: -494.3493042, 1629.8876953, -551.1360474, 1811.9309082, -2306.2792969, 2181.0236816
3: -702.1618652, 1774.0732422, -782.5770874, 1973.6967773, -2675.8586426, 2556.6503906
4: -828.9537354, 1656.8021240, -925.2904663, 1842.4506836, -2671.4042969, 2582.0925293

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0373133, upper bound: 1757.0677288
time: 0.91 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.0614236
time: 0.70 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -392.9946289, 1595.7901611, -392.9946289, 1595.7901611, -1988.7847900, 1988.7847900
1: -481.7033691, 1782.3461914, -481.7033691, 1782.3461914, -2264.0493164, 2264.0495605
2: -551.1360474, 1811.9309082, -551.1360474, 1811.9309082, -2363.0666504, 2363.0666504
3: -782.5770874, 1973.6967773, -782.5770874, 1973.6967773, -2756.2739258, 2756.2739258
4: -925.2904663, 1842.4506836, -925.2904663, 1842.4506836, -2767.7412109, 2767.7412109

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0373133, upper bound: 1757.1145273
time: 0.69 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.1044230
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.93 seconds
NS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0677288, upper bound: 1757.0373133
NS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.0354614
NS_B1_A2_A1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0677288, upper bound: 1757.0373171
NS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.0372694
NS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0373133, upper bound: 1757.0677288
NS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.0614236
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0373133, upper bound: 1757.1145273
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -1757.0354614, upper bound: 1757.1044230

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -367.8039856, 1492.2141113, -352.7528076, 1429.8717041, -1797.6754150, 1844.9669189
1: -450.7806702, 1666.2652588, -432.3590393, 1596.3819580, -2047.1625977, 2098.6242676
2: -516.1850586, 1693.7189941, -495.2670288, 1623.0344238, -2139.2194824, 2188.9860840
3: -732.7377930, 1846.9467773, -702.8384399, 1770.8116455, -2503.5493164, 2549.7849121
4: -867.3955688, 1722.8286133, -832.5136719, 1651.0700684, -2518.4655762, 2555.3422852

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043067, upper bound: 1757.1043455
time: 0.67 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043067, upper bound: 1757.1043487
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -368.8015747, 1504.0627441, -450.1930237, 1856.5698242, -2225.3710938, 1954.2557373
1: -451.9244080, 1679.6166992, -550.5508423, 2073.0170898, -2524.9411621, 2230.1669922
2: -518.0854492, 1707.0693359, -638.3911133, 2101.5507812, -2619.6362305, 2345.4604492
3: -735.2094116, 1859.8853760, -902.1766968, 2305.8959961, -3041.1054688, 2762.0617676
4: -871.0351562, 1735.8483887, -1082.5723877, 2138.3459473, -3009.3811035, 2818.4196777

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043349, upper bound: 1757.1044186
time: 0.72 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043349, upper bound: 1757.1044230
time: 0.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.22 seconds
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -1757.1043067, upper bound: 1757.1043455
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -1757.1043067, upper bound: 1757.1043487
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -1757.1043349, upper bound: 1757.1044186
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -1757.1043349, upper bound: 1757.1044230

## BFS NS instance: NS_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -352.7528076, 1429.8717041, -352.7528076, 1429.8717041, -1782.6245117, 1782.6245117
1: -432.3590393, 1596.3819580, -432.3590393, 1596.3819580, -2028.7409668, 2028.7409668
2: -495.2670288, 1623.0344238, -495.2670288, 1623.0344238, -2118.3015137, 2118.3015137
3: -702.8384399, 1770.8116455, -702.8384399, 1770.8116455, -2473.6499023, 2473.6499023
4: -832.5136719, 1651.0700684, -832.5136719, 1651.0700684, -2483.5837402, 2483.5837402

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A1_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0630647, upper bound: 1757.0887159
time: 0.65 seconds

## Relational analysis of NS_B2_A2_B1_A1_A2

### Relational analysis result of NS_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0682710
time: 1.35 seconds

## BFS NS instance: NS_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -450.1000977, 1856.1950684, -352.7528076, 1429.8717041, -1879.9718018, 2208.9477539
1: -550.4389038, 2072.6018066, -432.3590393, 1596.3819580, -2146.8208008, 2504.9604492
2: -638.2589111, 2101.1264648, -495.2670288, 1623.0344238, -2261.2934570, 2596.3933105
3: -901.9924316, 2305.4287109, -702.8384399, 1770.8116455, -2672.8039551, 3008.2670898
4: -1082.3427734, 2137.9162598, -832.5136719, 1651.0700684, -2733.4128418, 2970.4299316

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0630647, upper bound: 1757.0887159
time: 0.67 seconds

## Relational analysis of NS_B2_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0682710
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -352.7528076, 1429.8717041, -450.1930237, 1856.5698242, -2209.3227539, 1880.0646973
1: -432.3590393, 1596.3819580, -550.5508423, 2073.0170898, -2505.3759766, 2146.9328613
2: -495.2670288, 1623.0344238, -638.3911133, 2101.5507812, -2596.8178711, 2261.4255371
3: -702.8384399, 1770.8116455, -902.1766968, 2305.8959961, -3008.7343750, 2672.9882812
4: -832.5136719, 1651.0700684, -1082.5723877, 2138.3459473, -2970.8596191, 2733.6418457

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0943783, upper bound: 1757.0945315
time: 0.81 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0943783, upper bound: 1757.0943783
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -450.1930237, 1856.5698242, -450.1930237, 1856.5698242, -2306.7626953, 2306.7626953
1: -550.5508423, 2073.0170898, -550.5508423, 2073.0170898, -2623.5678711, 2623.5678711
2: -638.3911133, 2101.5507812, -638.3911133, 2101.5507812, -2739.9418945, 2739.9418945
3: -902.1766968, 2305.8959961, -902.1766968, 2305.8959961, -3208.0727539, 3208.0727539
4: -1082.5723877, 2138.3459473, -1082.5723877, 2138.3459473, -3220.9177246, 3220.9177246

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0621563, upper bound: 1757.0359613
time: 0.80 seconds

## Relational analysis of NS_B2_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343377, upper bound: 1757.0343377
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.74 seconds
NS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0630647, upper bound: 1757.0887159
NS_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0682710
NS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0630647, upper bound: 1757.0887159
NS_B2_A2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0682710
NS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0943783, upper bound: 1757.0945315
NS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0943783, upper bound: 1757.0943783
NS_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0621563, upper bound: 1757.0359613
NS_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -1757.0343377, upper bound: 1757.0343377

## BFS NS instance: NS_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -323.3041992, 1308.3963623, -337.6948853, 1368.7863770, -1692.0904541, 1646.0913086
1: -395.9123230, 1461.1677246, -413.8221130, 1528.1158447, -1924.0281982, 1874.9897461
2: -455.4548340, 1484.9721680, -474.2917480, 1553.6881104, -2009.1429443, 1959.2639160
3: -643.2018433, 1623.4108887, -672.5090332, 1694.8690186, -2338.0708008, 2295.9199219
4: -764.8485718, 1512.4916992, -796.8400269, 1580.4686279, -2345.3171387, 2309.3317871

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A1_A1_B1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1162557, upper bound: 1757.1162568
time: 0.81 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1162557, upper bound: 1757.1177106
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -438.1200867, 1804.6134033, -337.6948853, 1368.7863770, -1806.9063721, 2142.3083496
1: -535.1567993, 2015.0118408, -413.8221130, 1528.1158447, -2063.2727051, 2428.8337402
2: -621.6227417, 2042.1135254, -474.2917480, 1553.6881104, -2175.3105469, 2516.4050293
3: -876.5314941, 2243.6308594, -672.5090332, 1694.8690186, -2571.4003906, 2916.1398926
4: -1053.8990479, 2079.1586914, -796.8400269, 1580.4686279, -2634.3676758, 2875.9987793

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0664138
time: 0.97 seconds

## Relational analysis of NS_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0682710
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -342.8920593, 1390.2570801, -418.2335815, 1726.7127686, -2069.6044922, 1808.4904785
1: -420.2868347, 1552.1630859, -511.6340027, 1928.8596191, -2349.1464844, 2063.7971191
2: -481.5502625, 1578.0948486, -592.1152344, 1955.3981934, -2436.9482422, 2170.2099609
3: -683.2750854, 1721.9652100, -837.6755371, 2141.4594727, -2824.7346191, 2559.6403809
4: -809.6408081, 1605.3092041, -1002.3061523, 1988.3162842, -2797.9570312, 2607.6152344

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0945206, upper bound: 1757.0943783
time: 0.69 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0945206, upper bound: 1757.0943783
time: 1.06 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -350.9931641, 1422.5258789, -443.2770081, 1826.9315186, -2177.9245605, 1865.8028564
1: -430.2149048, 1588.1575928, -542.2296143, 2039.9180908, -2470.1328125, 2130.3869629
2: -492.7920227, 1614.7390137, -628.5104370, 2068.2482910, -2561.0402832, 2243.2495117
3: -699.3253784, 1761.7321777, -888.3466797, 2269.0156250, -2968.3410645, 2650.0788574
4: -828.3157349, 1642.6568604, -1065.5307617, 2104.4018555, -2932.7175293, 2708.1875000

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1000828, upper bound: 1757.0877946
time: 0.73 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0642202, upper bound: 1757.0352180
time: 0.76 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0654202, upper bound: 1757.0357460
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.03 seconds
NS_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.1162557, upper bound: 1757.1162568
NS_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.1162557, upper bound: 1757.1177106
NS_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0664138
NS_B2_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.0361442, upper bound: 1757.0682710
NS_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.0945206, upper bound: 1757.0943783
NS_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.0945206, upper bound: 1757.0943783
NS_B2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.0642202, upper bound: 1757.0352180
NS_B2_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.03
Output dim: 0, lower bound: -1757.0654202, upper bound: 1757.0357460

## BFS NS instance: NS_B2_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -323.3041992, 1308.3963623, -323.3041992, 1308.3963623, -1631.7005615, 1631.7005615
1: -395.9123230, 1461.1677246, -395.9123230, 1461.1677246, -1857.0799561, 1857.0799561
2: -455.4548340, 1484.9721680, -455.4548340, 1484.9721680, -1940.4270020, 1940.4270020
3: -643.2018433, 1623.4108887, -643.2018433, 1623.4108887, -2266.6127930, 2266.6127930
4: -764.8485718, 1512.4916992, -764.8485718, 1512.4916992, -2277.3403320, 2277.3403320

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0416400, upper bound: 1757.0734217
time: 0.75 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0353386, upper bound: 1757.0704575
time: 0.69 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -323.3041992, 1308.3963623, -342.8786621, 1390.2818604, -1713.5859375, 1651.2750244
1: -395.9123230, 1461.1677246, -420.3764038, 1552.1739502, -1948.0863037, 1881.5439453
2: -455.4548340, 1484.9721680, -481.4416199, 1578.3128662, -2033.7675781, 1966.4138184
3: -643.2018433, 1623.4108887, -683.2790527, 1721.8658447, -2365.0676270, 2306.6899414
4: -764.8485718, 1512.4916992, -809.0892334, 1605.8089600, -2370.6574707, 2321.5810547

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1141053, upper bound: 1757.1174076
time: 0.80 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137647, upper bound: 1757.1153307
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -309.5700378, 1259.4353027, -418.2335815, 1726.7127686, -2036.2827148, 1677.6687012
1: -379.7421265, 1406.8647461, -511.6340027, 1928.8596191, -2308.6018066, 1918.4987793
2: -433.3220215, 1429.6942139, -592.1152344, 1955.3981934, -2388.7202148, 2021.8094482
3: -616.7457275, 1556.0572510, -837.6755371, 2141.4594727, -2758.2050781, 2393.7329102
4: -726.5832520, 1453.5422363, -1002.3061523, 1988.3162842, -2714.8994141, 2455.8481445

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926124, upper bound: 1757.0925858
time: 0.68 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
time: 0.69 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -349.7909546, 1417.5225830, -418.2335815, 1726.7127686, -2076.5034180, 1835.7561035
1: -428.7479553, 1582.5559082, -511.6340027, 1928.8596191, -2357.6076660, 2094.1899414
2: -491.1026001, 1609.0854492, -592.1152344, 1955.3981934, -2446.5000000, 2201.2004395
3: -696.9265137, 1755.5729980, -837.6755371, 2141.4594727, -2838.3857422, 2593.2485352
4: -825.4575806, 1636.9227295, -1002.3061523, 1988.3162842, -2813.7739258, 2639.2285156

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926124, upper bound: 1757.0925858
time: 0.77 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.48 seconds
NS_B2_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1141053, upper bound: 1757.1174076
NS_B2_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1137647, upper bound: 1757.1153307
NS_B2_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0926124, upper bound: 1757.0925858
NS_B2_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
NS_B2_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0926124, upper bound: 1757.0925858
NS_B2_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -323.3041992, 1308.3963623, -342.2456055, 1387.6904297, -1710.9943848, 1650.6419678
1: -395.9123230, 1461.1677246, -419.6022339, 1549.2843018, -1945.1966553, 1880.7698975
2: -455.4548340, 1484.9721680, -480.5501404, 1575.3740234, -2030.8287354, 1965.5222168
3: -643.2018433, 1623.4108887, -682.0135498, 1718.6522217, -2361.8535156, 2305.4243164
4: -764.8485718, 1512.4916992, -807.5834351, 1602.8165283, -2367.6650391, 2320.0751953

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1126236, upper bound: 1757.1014361
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1141053, upper bound: 1757.1174076
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -323.3041992, 1308.3963623, -349.8536682, 1420.0977783, -1743.4017334, 1658.2500000
1: -395.9123230, 1461.1677246, -428.9212036, 1585.6981201, -1981.6103516, 1890.0888672
2: -455.4548340, 1484.9721680, -491.2429504, 1612.0653076, -2067.5200195, 1976.2149658
3: -643.2018433, 1623.4108887, -697.3499146, 1758.7243652, -2401.9262695, 2320.7607422
4: -764.8485718, 1512.4916992, -825.5804443, 1640.3387451, -2405.1872559, 2338.0722656

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1123065, upper bound: 1757.0987751
time: 0.72 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137647, upper bound: 1757.1153307
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -308.8511658, 1256.4797363, -418.2335815, 1726.7127686, -2035.5638428, 1674.7131348
1: -378.8622742, 1403.5659180, -511.6340027, 1928.8596191, -2307.7219238, 1915.1999512
2: -432.3126221, 1426.3400879, -592.1152344, 1955.3981934, -2387.7104492, 2018.4553223
3: -615.3094482, 1552.4041748, -837.6755371, 2141.4594727, -2756.7690430, 2390.0795898
4: -724.8827515, 1450.1318359, -1002.3061523, 1988.3162842, -2713.1989746, 2452.4379883

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -316.2204590, 1287.8253174, -418.2335815, 1726.7127686, -2042.9328613, 1706.0588379
1: -387.8865662, 1438.6445312, -511.6340027, 1928.8596191, -2316.7460938, 1950.2785645
2: -442.7817993, 1461.8300781, -592.1152344, 1955.3981934, -2398.1796875, 2053.9453125
3: -630.2246704, 1591.4664307, -837.6755371, 2141.4594727, -2771.6838379, 2429.1418457
4: -742.6939697, 1486.4067383, -1002.3061523, 1988.3162842, -2731.0102539, 2488.7126465

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -349.0071716, 1414.3284912, -418.2335815, 1726.7127686, -2075.7197266, 1832.5620117
1: -427.7903137, 1578.9929199, -511.6340027, 1928.8596191, -2356.6499023, 2090.6269531
2: -490.0046997, 1605.4631348, -592.1152344, 1955.3981934, -2445.4028320, 2197.5783691
3: -695.3643188, 1751.6174316, -837.6755371, 2141.4594727, -2836.8234863, 2589.2924805
4: -823.6053467, 1633.2362061, -1002.3061523, 1988.3162842, -2811.9216309, 2635.5419922

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
time: 0.68 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -356.6078186, 1446.7607422, -418.2335815, 1726.7127686, -2083.3205566, 1864.9941406
1: -437.1392212, 1615.3048096, -511.6340027, 1928.8596191, -2365.9987793, 2126.9384766
2: -500.7145996, 1642.1793213, -592.1152344, 1955.3981934, -2456.1123047, 2234.2939453
3: -710.7427979, 1791.6578369, -837.6755371, 2141.4594727, -2852.2021484, 2629.3332520
4: -841.6430664, 1670.5660400, -1002.3061523, 1988.3162842, -2829.9594727, 2672.8718262

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
time: 0.98 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
time: 1.15 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.75 seconds
NS_B2_A2_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.1126236, upper bound: 1757.1014361
NS_B2_A2_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.1141053, upper bound: 1757.1174076
NS_B2_A2_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.1123065, upper bound: 1757.0987751
NS_B2_A2_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.1137647, upper bound: 1757.1153307
NS_B2_A2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
NS_B2_A2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
NS_B2_A2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858
NS_B2_A2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.75
Output dim: 0, lower bound: -1757.0925865, upper bound: 1757.0925858

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -306.1368713, 1235.6409912, -310.4482422, 1251.7478027, -1557.8846436, 1546.0892334
1: -374.7603149, 1379.8162842, -380.0846252, 1397.7058105, -1772.4659424, 1759.9007568
2: -431.6030579, 1402.6842041, -434.8385925, 1421.8902588, -1853.4932861, 1837.5225830
3: -608.6734619, 1534.6442871, -616.4312744, 1550.5549316, -2159.2282715, 2151.0756836
4: -725.0646362, 1429.0267334, -729.6229858, 1447.2427979, -2172.3073730, 2158.6496582

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1003415, upper bound: 1757.1008540
time: 0.72 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1003415, upper bound: 1757.1014361
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -322.7723999, 1306.1910400, -340.5453491, 1380.5937500, -1703.3659668, 1646.7363281
1: -395.2604370, 1458.7065430, -417.5184937, 1541.3587646, -1936.6191406, 1876.2250977
2: -454.7073059, 1482.4730225, -478.1556091, 1567.3457031, -2022.0528564, 1960.6286621
3: -642.1334229, 1620.6789551, -678.5908203, 1709.8946533, -2352.0278320, 2299.2695312
4: -763.5802612, 1509.9511719, -803.5272217, 1594.6588135, -2358.2390137, 2313.4780273

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_B1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1134277, upper bound: 1757.1143316
time: 0.73 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B1_B2_B2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1141053, upper bound: 1757.1174076
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -306.1368713, 1235.6409912, -318.9522705, 1288.0742188, -1594.2110596, 1554.5930176
1: -374.7603149, 1379.8162842, -390.5514832, 1438.3760986, -1813.1362305, 1770.3675537
2: -431.6030579, 1402.6842041, -446.8404236, 1462.9962158, -1894.5992432, 1849.5245361
3: -608.6734619, 1534.6442871, -633.5932617, 1595.4912109, -2204.1645508, 2168.2375488
4: -725.0646362, 1429.0267334, -749.9299316, 1488.9240723, -2213.9887695, 2178.9562988

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0998598, upper bound: 1757.0981803
time: 0.77 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1123065, upper bound: 1757.0987751
time: 0.76 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1123065, upper bound: 1757.0987751
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_B1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -322.7723999, 1306.1910400, -348.1272278, 1412.8995361, -1735.6717529, 1654.3181152
1: -395.2604370, 1458.7065430, -426.8038330, 1577.6772461, -1972.9376221, 1885.5103760
2: -454.7073059, 1482.4730225, -488.8119507, 1603.9095459, -2058.6162109, 1971.2849121
3: -642.1334229, 1620.6789551, -693.8764038, 1749.8337402, -2391.9667969, 2314.5554199
4: -763.5802612, 1509.9511719, -821.4667358, 1632.0520020, -2395.6323242, 2331.4177246

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1009243, upper bound: 1757.1143540
time: 0.75 seconds

## Relational analysis of NS_B2_A2_B1_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_B2_A2_B1_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1009243, upper bound: 1757.1153307
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -349.0071716, 1414.3284912, -417.4485779, 1723.4350586, -2072.4421387, 1831.7770996
1: -427.7903137, 1578.9929199, -510.6698914, 1925.2065430, -2352.9968262, 2089.6628418
2: -490.0046997, 1605.4631348, -591.0094604, 1951.6784668, -2441.6831055, 2196.4726562
3: -695.3643188, 1751.6174316, -836.0942383, 2137.4243164, -2832.7883301, 2587.7109375
4: -823.6053467, 1633.2362061, -1000.4394531, 1984.5471191, -2808.1520996, 2633.6750488

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -349.0071716, 1414.3284912, -424.9657288, 1754.8616943, -2103.8688965, 1839.2941895
1: -427.7903137, 1578.9929199, -519.9470215, 1960.3907471, -2388.1811523, 2098.9399414
2: -490.0046997, 1605.4631348, -601.5819092, 1987.6704102, -2477.6750488, 2207.0449219
3: -695.3643188, 1751.6174316, -851.2218018, 2176.1494141, -2871.5134277, 2602.8383789
4: -823.6053467, 1633.2362061, -1018.1728516, 2021.1182861, -2844.7233887, 2651.4086914

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -356.6078186, 1446.7607422, -417.4485779, 1723.4350586, -2080.0427246, 1864.2092285
1: -437.1392212, 1615.3048096, -510.6698914, 1925.2065430, -2362.3457031, 2125.9746094
2: -500.7145996, 1642.1793213, -591.0094604, 1951.6784668, -2452.3925781, 2233.1882324
3: -710.7427979, 1791.6578369, -836.0942383, 2137.4243164, -2848.1669922, 2627.7517090
4: -841.6430664, 1670.5660400, -1000.4394531, 1984.5471191, -2826.1899414, 2671.0046387

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -356.6078186, 1446.7607422, -424.9657288, 1754.8616943, -2111.4694824, 1871.7264404
1: -437.1392212, 1615.3048096, -519.9470215, 1960.3907471, -2397.5300293, 2135.2514648
2: -500.7145996, 1642.1793213, -601.5819092, 1987.6704102, -2488.3845215, 2243.7607422
3: -710.7427979, 1791.6578369, -851.2218018, 2176.1494141, -2886.8920898, 2642.8791504
4: -841.6430664, 1670.5660400, -1018.1728516, 2021.1182861, -2862.7612305, 2688.7382812

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.56 + 125.54 = 129.10 seconds
