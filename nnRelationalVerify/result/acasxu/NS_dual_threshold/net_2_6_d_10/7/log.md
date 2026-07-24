## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 2585.384444397015


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-997.0294189, 1909.5061035, -997.0294189, 1909.5061035, -2906.5351562, 2906.5351562)
1: (-191.0864868, 246.2004242, -191.0864868, 246.2004242, -437.2869263, 437.2869263)
2: (-145.9371338, 327.1642761, -145.9371338, 327.1642761, -473.1014099, 473.1014099)
3: (-142.7629242, 425.7612000, -142.7629242, 425.7612000, -568.5241089, 568.5241089)
4: (-125.2214203, 411.2394104, -125.2214203, 411.2394104, -536.4608154, 536.4608154)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.35 + 1.92 = 4.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2585.4102985, upper bound: 2585.4102985

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4102076, upper bound: 2585.4102081
time: 0.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4102131, upper bound: 2585.4102114
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.52 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -2585.4102076, upper bound: 2585.4102081
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -2585.4102131, upper bound: 2585.4102114

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -960.9322510, 1824.3105469, -978.5440063, 1868.2806396, -2829.2126465, 2802.8544922
1: -181.1123199, 235.6690674, -186.7179718, 241.1905060, -422.3027649, 422.3870239
2: -139.4877777, 314.8420105, -142.9165344, 320.7892151, -460.2769775, 457.7585449
3: -137.2162170, 408.6289673, -139.9641113, 417.2837830, -554.4999390, 548.5930786
4: -119.7274399, 395.8403931, -122.6212234, 403.3000488, -523.0274658, 518.4616089

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101083, upper bound: 2585.4098431
time: 0.73 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097999, upper bound: 2585.4098414
time: 0.70 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -990.9934082, 1898.7631836, -997.0294189, 1909.5061035, -2900.4995117, 2895.7924805
1: -190.0336304, 244.7710114, -191.0864868, 246.2004242, -436.2339783, 435.8574524
2: -145.0904846, 325.0429382, -145.9371338, 327.1642761, -472.2547607, 470.9800720
3: -141.8780975, 423.2347717, -142.7629242, 425.7612000, -567.6392822, 565.9975586
4: -124.5006714, 408.5954590, -125.2214203, 411.2394104, -535.7399902, 533.8168335

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101189, upper bound: 2585.4102051
time: 0.66 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101189, upper bound: 2585.4101190
time: 0.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.68 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -2585.4101083, upper bound: 2585.4098431
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -2585.4097999, upper bound: 2585.4098414
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -2585.4101189, upper bound: 2585.4102051
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -2585.4101189, upper bound: 2585.4101190

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -960.9322510, 1824.3105469, -958.9455566, 1830.9945068, -2791.9267578, 2783.2558594
1: -181.1123199, 235.6690674, -182.9884644, 236.3887177, -417.5010376, 418.6575012
2: -139.4877777, 314.8420105, -140.0497131, 314.1820068, -453.6697693, 454.8917236
3: -137.2162170, 408.6289673, -137.1166077, 408.8638916, -546.0800781, 545.7456055
4: -119.7274399, 395.8403931, -120.1583328, 395.0694275, -514.7968750, 515.9987183

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099987, upper bound: 2585.4098404
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099972, upper bound: 2585.4096322
time: 0.74 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -960.9322510, 1824.3105469, -1001.2144165, 1908.6560059, -2869.5881348, 2825.5246582
1: -181.1123199, 235.6690674, -190.5679626, 246.5025330, -427.6148682, 426.2370300
2: -139.4877777, 314.8420105, -146.1399689, 326.7961426, -466.2839050, 460.9819946
3: -137.2162170, 408.6289673, -143.2377014, 426.3121338, -563.5283203, 551.8666992
4: -119.7274399, 395.8403931, -125.3912811, 410.8598328, -530.5872192, 521.2316895

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097584, upper bound: 2585.4098379
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097568, upper bound: 2585.4096295
time: 0.83 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -1012.3721924, 1932.6853027, -993.3973389, 1901.8802490, -2914.2521973, 2926.0825195
1: -193.4377747, 249.3907318, -190.3289337, 245.2494659, -438.6872559, 439.7196655
2: -147.9088745, 331.2546082, -145.3644867, 325.9311523, -473.8399963, 476.6190796
3: -144.9719391, 431.6279297, -142.2198944, 424.1388550, -569.1107178, 573.8477173
4: -126.9188080, 416.3009033, -124.7265701, 409.7063599, -536.6251221, 541.0274658

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100027, upper bound: 2585.4097912
time: 0.68 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097624, upper bound: 2585.4097898
time: 0.77 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -984.6502686, 1887.7196045, -997.0294189, 1909.5061035, -2894.1562500, 2884.7485352
1: -188.9507294, 243.3076782, -191.0864868, 246.2004242, -435.1511536, 434.3941650
2: -144.2333374, 323.0198364, -145.9371338, 327.1642761, -471.3976135, 468.9569702
3: -140.9723969, 420.6856384, -142.7629242, 425.7612000, -566.7335815, 563.4485474
4: -123.7614365, 406.0218811, -125.2214203, 411.2394104, -535.0008545, 531.2432861

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100037, upper bound: 2585.4097673
time: 0.67 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097636, upper bound: 2585.4097634
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.63 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4099987, upper bound: 2585.4098404
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4099972, upper bound: 2585.4096322
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4097584, upper bound: 2585.4098379
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4097568, upper bound: 2585.4096295
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4100027, upper bound: 2585.4097912
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4097624, upper bound: 2585.4097898
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4100037, upper bound: 2585.4097673
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -2585.4097636, upper bound: 2585.4097634

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -955.4187622, 1823.9759521, -2790.2072754, 2788.6186523
1: -182.0998230, 236.8301697, -182.2559052, 235.4848480, -417.5845642, 419.0860291
2: -140.2225494, 316.3553467, -139.5117493, 313.0054321, -453.2279663, 455.8670654
3: -138.0129700, 410.7770386, -136.5889893, 407.3274536, -545.3404541, 547.3660278
4: -120.3594055, 397.6459351, -119.6962738, 393.5971985, -513.9566040, 517.3422241

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098343
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098404
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -958.9455566, 1830.9945068, -2785.2807617, 2770.6120605
1: -179.8823242, 234.0388947, -182.9884644, 236.3887177, -416.2710571, 417.0272522
2: -138.5360565, 312.7053833, -140.0497131, 314.1820068, -452.7180481, 452.7550964
3: -136.2803802, 405.7934875, -137.1166077, 408.8638916, -545.1442261, 542.9099731
4: -118.9099197, 393.1340637, -120.1583328, 395.0694275, -513.9793701, 513.2924194

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096265
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096323
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -997.6569824, 1901.5863037, -2867.8173828, 2830.8571777
1: -182.0998230, 236.8301697, -189.8340302, 245.5929565, -427.6927490, 426.6641846
2: -140.2225494, 316.3553467, -145.5966339, 325.6049194, -465.8274536, 461.9519653
3: -138.0129700, 410.7770386, -142.7048798, 424.7533875, -562.7663574, 553.4819336
4: -120.3594055, 397.6459351, -124.9240112, 409.3783569, -529.7377930, 522.5699463

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098318
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098373
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -1001.2144165, 1908.6560059, -2862.9418945, 2812.8808594
1: -179.8823242, 234.0388947, -190.5679626, 246.5025330, -426.3848572, 424.6068420
2: -138.5360565, 312.7053833, -146.1399689, 326.7961426, -465.3321838, 458.8453369
3: -136.2803802, 405.7934875, -143.2377014, 426.3121338, -562.5925293, 549.0310059
4: -118.9099197, 393.1340637, -125.3912811, 410.8598328, -529.7697144, 518.5252686

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096226
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096295
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1012.3721924, 1932.6853027, -973.6421509, 1863.9254150, -2876.2976074, 2906.3273926
1: -193.4377747, 249.3907318, -186.5438232, 240.3875885, -433.8253784, 435.9345703
2: -147.9088745, 331.2546082, -142.4597321, 319.2463379, -467.1551819, 473.7143555
3: -144.9719391, 431.6279297, -139.3439331, 415.6249390, -560.5968628, 570.9718018
4: -126.9188080, 416.3009033, -122.2217712, 401.3868103, -528.3054810, 538.5227051

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097870
time: 0.79 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
time: 0.90 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1012.3721924, 1932.6853027, -1015.2727661, 1940.1262207, -2952.4980469, 2947.9577637
1: -193.4377747, 249.3907318, -194.0003662, 250.3372192, -443.7749939, 443.3911133
2: -147.9088745, 331.2546082, -148.4544678, 331.7157898, -479.6246338, 479.7090454
3: -144.9719391, 431.6279297, -145.3659821, 432.7805176, -577.7523804, 576.9938965
4: -126.9188080, 416.3009033, -127.3560410, 416.9845886, -543.9033813, 543.6569214

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097624, upper bound: 2585.4097898
time: 0.73 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097624, upper bound: 2585.4097898
time: 0.68 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -984.6502686, 1887.7196045, -977.2257080, 1871.5139160, -2856.1640625, 2864.9450684
1: -188.9507294, 243.3076782, -187.3010864, 241.3262329, -430.2769775, 430.6087341
2: -144.2333374, 323.0198364, -143.0208588, 320.4586792, -464.6920166, 466.0407104
3: -140.9723969, 420.6856384, -139.8863068, 417.2209167, -558.1931763, 560.5719604
4: -123.7614365, 406.0218811, -122.7130432, 402.9029541, -526.6643677, 528.7349243

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097601
time: 0.79 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097635
time: 0.87 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -984.6502686, 1887.7196045, -1018.9093018, 1947.7835693, -2932.4338379, 2906.6284180
1: -188.9507294, 243.3076782, -194.7627869, 251.2817993, -440.2325439, 438.0704651
2: -144.2333374, 323.0198364, -149.0220337, 332.9416809, -477.1750183, 472.0418701
3: -140.9723969, 420.6856384, -145.9150543, 434.4020386, -575.3742676, 566.6007080
4: -123.7614365, 406.0218811, -127.8521500, 418.5213928, -542.2828369, 533.8740234

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097636, upper bound: 2585.4097622
time: 0.72 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097636, upper bound: 2585.4097622
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.88 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098343
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098404
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096265
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096323
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098318
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098373
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096226
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096295
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097870
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097624, upper bound: 2585.4097898
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097624, upper bound: 2585.4097898
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097601
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097635
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097636, upper bound: 2585.4097622
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097636, upper bound: 2585.4097622

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -937.1047363, 1779.4067383, -2745.6381836, 2770.3046875
1: -182.0998230, 236.8301697, -176.6248169, 229.8726501, -411.9724731, 413.4549866
2: -140.2225494, 316.3553467, -136.0071716, 306.9665222, -447.1890869, 452.3625183
3: -138.0129700, 410.7770386, -133.7311707, 398.4748535, -536.4877930, 544.5081787
4: -120.3594055, 397.6459351, -116.7290802, 386.0252991, -506.3846741, 514.3749390

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097244, upper bound: 2585.4098356
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097244, upper bound: 2585.4098356
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -967.5244141, 1852.9454346, -2819.1767578, 2800.7243652
1: -182.0998230, 236.8301697, -185.4743958, 238.9097748, -421.0095825, 422.3045654
2: -140.2225494, 316.3553467, -141.5873871, 317.0906677, -457.3132324, 457.9427185
3: -138.0129700, 410.7770386, -138.4504395, 412.9884644, -551.0013428, 549.2273560
4: -120.3594055, 397.6459351, -121.4871521, 398.6987000, -519.0581055, 519.1329956

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068029, upper bound: 2585.4097535
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098343
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -940.8821411, 1786.6137695, -2740.8999023, 2752.5485840
1: -179.8823242, 234.0388947, -177.3397827, 230.8084869, -410.6907959, 411.3786621
2: -138.5360565, 312.7053833, -136.5652008, 308.2008057, -446.7368469, 449.2705688
3: -136.2803802, 405.7934875, -134.2920074, 400.0929260, -536.3732300, 540.0854492
4: -118.9099197, 393.1340637, -117.2141190, 387.5705872, -506.4804993, 510.3481750

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096265
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096264
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -971.1600342, 1860.6889648, -2814.9750977, 2782.8266602
1: -179.8823242, 234.0388947, -186.2384338, 239.8831482, -419.7654724, 420.2772827
2: -138.5360565, 312.7053833, -142.1675415, 318.3192749, -456.8553467, 454.8728943
3: -136.2803802, 405.7934875, -138.9995880, 414.6497803, -550.9301758, 544.7930298
4: -118.9099197, 393.1340637, -121.9870453, 400.2417603, -519.1516724, 515.1210938

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068014, upper bound: 2585.4094694
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097223, upper bound: 2585.4096305
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -982.2663574, 1862.2283936, -2828.4597168, 2815.4665527
1: -182.0998230, 236.8301697, -184.8478394, 240.6604156, -422.7602234, 421.6779480
2: -140.2225494, 316.3553467, -142.5335999, 320.1076355, -460.3302002, 458.8889465
3: -138.0129700, 410.7770386, -140.3580475, 417.0577698, -555.0707397, 551.1350708
4: -120.3594055, 397.6459351, -122.3712540, 402.5979614, -522.9573975, 520.0171509

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098323
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098316
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -1009.1226196, 1928.9887695, -2895.2202148, 2842.3225098
1: -182.0998230, 236.8301697, -192.9198761, 248.8334808, -430.9332886, 429.7500610
2: -140.2225494, 316.3553467, -147.5729675, 329.5365295, -469.7590942, 463.9283142
3: -138.0129700, 410.7770386, -144.4653015, 430.1126709, -568.1256104, 555.2423096
4: -120.3594055, 397.6459351, -126.6131210, 414.2673340, -534.6267090, 524.2590332

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068827, upper bound: 2585.4097118
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098379
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -986.2863159, 1869.9299316, -2824.2158203, 2797.9523926
1: -179.8823242, 234.0388947, -185.6130371, 241.6552277, -421.5375366, 419.6518555
2: -138.5360565, 312.7053833, -143.1280365, 321.4186707, -459.9547119, 455.8334045
3: -136.2803802, 405.7934875, -140.9549103, 418.7827454, -555.0629883, 546.7484131
4: -118.9099197, 393.1340637, -122.8881760, 404.2409363, -523.1508179, 516.0222168

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096233
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096228
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -1012.8219604, 1936.8083496, -2891.0947266, 2824.4885254
1: -179.8823242, 234.0388947, -193.6926422, 249.8078156, -429.6901245, 427.7314758
2: -138.5360565, 312.7053833, -148.1551361, 330.7892761, -469.3252869, 460.8605042
3: -136.2803802, 405.7934875, -145.0234528, 431.7693176, -568.0496826, 550.8168945
4: -118.9099197, 393.1340637, -127.1176224, 415.8401184, -534.7500000, 520.2517090

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068801, upper bound: 2585.4094267
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096228, upper bound: 2585.4096284
time: 0.69 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1012.3721924, 1932.6853027, -937.1047363, 1779.4067383, -2791.7788086, 2869.7897949
1: -193.4377747, 249.3907318, -176.6248169, 229.8726501, -423.3104248, 426.0155334
2: -147.9088745, 331.2546082, -136.0071716, 306.9665222, -454.8753662, 467.2617493
3: -144.9719391, 431.6279297, -133.7311707, 398.4748535, -543.4467773, 565.3591309
4: -126.9188080, 416.3009033, -116.7290802, 386.0252991, -512.9440918, 533.0299072

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097868
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
time: 0.65 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1012.3721924, 1932.6853027, -967.5244141, 1852.9454346, -2865.3176270, 2900.2094727
1: -193.4377747, 249.3907318, -185.4743958, 238.9097748, -432.3475342, 434.8651123
2: -147.9088745, 331.2546082, -141.5873871, 317.0906677, -464.9995117, 472.8419800
3: -144.9719391, 431.6279297, -138.4504395, 412.9884644, -557.9601440, 570.0782471
4: -126.9188080, 416.3009033, -121.4871521, 398.6987000, -525.6174316, 537.7879639

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097868
time: 0.75 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
time: 0.74 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -993.6475220, 1896.5800781, -1015.2727661, 1940.1262207, -2933.7734375, 2911.8527832
1: -189.7981720, 244.7791901, -194.0003662, 250.3372192, -440.1353760, 438.7795410
2: -145.1436310, 324.8939209, -148.4544678, 331.7157898, -476.8594360, 473.3483887
3: -142.2358246, 423.5339661, -145.3659821, 432.7805176, -575.0163574, 568.8999023
4: -124.5412140, 408.3838196, -127.3560410, 416.9845886, -541.5257568, 535.7398682

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097830
time: 0.72 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097837
time: 0.72 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1029.8892822, 1963.0853271, -1015.2727661, 1940.1262207, -2970.0151367, 2978.3574219
1: -196.2228851, 253.4745789, -194.0003662, 250.3372192, -446.5601196, 447.4749451
2: -150.3553009, 335.8014221, -148.4544678, 331.7157898, -482.0710449, 484.2558594
3: -147.4473724, 438.5150452, -145.3659821, 432.7805176, -580.2279053, 583.8810425
4: -128.9989624, 422.1042175, -127.3560410, 416.9845886, -545.9835205, 549.4602051

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097831
time: 0.75 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097829
time: 0.82 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -984.6502686, 1887.7196045, -940.8821411, 1786.6137695, -2771.2639160, 2828.6015625
1: -188.9507294, 243.3076782, -177.3397827, 230.8084869, -419.7592163, 420.6474609
2: -144.2333374, 323.0198364, -136.5652008, 308.2008057, -452.4341431, 459.5849915
3: -140.9723969, 420.6856384, -134.2920074, 400.0929260, -541.0652466, 554.9776611
4: -123.7614365, 406.0218811, -117.2141190, 387.5705872, -511.3319702, 523.2360229

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097285, upper bound: 2585.4097599
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097588
time: 0.69 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -984.6502686, 1887.7196045, -971.1600342, 1860.6889648, -2845.3391113, 2858.8796387
1: -188.9507294, 243.3076782, -186.2384338, 239.8831482, -428.8338623, 429.5460815
2: -144.2333374, 323.0198364, -142.1675415, 318.3192749, -462.5526123, 465.1873474
3: -140.9723969, 420.6856384, -138.9995880, 414.6497803, -555.6221313, 559.6852417
4: -123.7614365, 406.0218811, -121.9870453, 400.2417603, -524.0031738, 528.0089111

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097654
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097635
time: 0.87 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -964.6633911, 1849.4851074, -1018.9093018, 1947.7835693, -2912.4470215, 2868.3942871
1: -185.1419525, 238.3877411, -194.7627869, 251.2817993, -436.4237671, 433.1505127
2: -141.2984924, 316.2402954, -149.0220337, 332.9416809, -474.2401428, 465.2623291
3: -138.0811615, 412.0292664, -145.9150543, 434.4020386, -572.4832153, 557.9443359
4: -121.2370834, 397.6160278, -127.8521500, 418.5213928, -539.7584229, 525.4680786

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4091896, upper bound: 2585.4095999
time: 0.72 seconds

## Relational analysis of NS_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4091862, upper bound: 2585.4091845
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1006.0653076, 1925.2583008, -1018.9093018, 1947.7835693, -2953.8488770, 2944.1669922
1: -192.5580902, 248.2636566, -194.7627869, 251.2817993, -443.8399048, 443.0264282
2: -147.2548065, 328.6759949, -149.0220337, 332.9416809, -480.1964722, 477.6980286
3: -144.0786743, 429.0533142, -145.9150543, 434.4020386, -578.4806519, 574.9683838
4: -126.3390961, 413.1736755, -127.8521500, 418.5213928, -544.8604736, 541.0258179

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096293, upper bound: 2585.4097561
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096293, upper bound: 2585.4097615
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.85 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097244, upper bound: 2585.4098356
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097244, upper bound: 2585.4098356
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4068029, upper bound: 2585.4097535
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098343
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096265
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096264
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4068014, upper bound: 2585.4094694
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097223, upper bound: 2585.4096305
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098323
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098316
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4068827, upper bound: 2585.4097118
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098379
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096233
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096228
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4068801, upper bound: 2585.4094267
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096228, upper bound: 2585.4096284
NS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097868
NS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
NS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097868
NS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
NS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097830
NS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097837
NS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097831
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097829
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097285, upper bound: 2585.4097599
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097588
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097654
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4097278, upper bound: 2585.4097635
NS_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4091896, upper bound: 2585.4095999
NS_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4091862, upper bound: 2585.4091845
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096293, upper bound: 2585.4097561
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -2585.4096293, upper bound: 2585.4097615

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -947.2660522, 1797.3228760, -937.1047363, 1779.4067383, -2726.6728516, 2734.4277344
1: -178.4807892, 232.2029572, -176.6248169, 229.8726501, -408.3534241, 408.8277588
2: -137.4472046, 310.1622620, -136.0071716, 306.9665222, -444.4136963, 446.1694336
3: -135.2370911, 402.6688232, -133.7311707, 398.4748535, -533.7119141, 536.3999634
4: -117.9695969, 389.9200134, -116.7290802, 386.0252991, -503.9948425, 506.6491089

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068025, upper bound: 2585.4097088
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098354
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -989.3084106, 1873.9550781, -937.1047363, 1779.4067383, -2768.7150879, 2811.0595703
1: -186.0719604, 242.2192078, -176.6248169, 229.8726501, -415.9446106, 418.8439941
2: -143.4731293, 322.3971863, -136.0071716, 306.9665222, -450.4396362, 458.4043579
3: -141.3676910, 419.8358154, -133.7311707, 398.4748535, -539.8425293, 553.5668945
4: -123.1527328, 405.3808289, -116.7290802, 386.0252991, -509.1780396, 522.1097412

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068029, upper bound: 2585.4097088
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098358
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -941.8323975, 1782.5511475, -892.7998047, 1707.9658203, -2649.7983398, 2675.3510742
1: -176.9412689, 230.4274750, -170.3612976, 220.3603821, -397.3016357, 400.7887573
2: -136.4774017, 308.0725708, -130.8826752, 291.8146362, -428.2920532, 438.9552612
3: -134.4552307, 399.9111023, -128.6929626, 381.2327881, -515.6879883, 528.6040039
4: -117.1305695, 387.2534180, -112.3672485, 366.6970825, -483.8276062, 499.6206665

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4077118, upper bound: 2585.4096895
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4073584, upper bound: 2585.4096037
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075149, upper bound: 2585.4096120
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -964.4354858, 1847.2637939, -2813.4951172, 2797.6357422
1: -182.0998230, 236.8301697, -184.9117889, 238.1691132, -420.2688904, 421.7418823
2: -140.2225494, 316.3553467, -141.1489868, 316.1160889, -456.3386230, 457.5043030
3: -138.0129700, 410.7770386, -138.0212555, 411.7046204, -549.7175903, 548.7982788
4: -120.3594055, 397.6459351, -121.1162796, 397.4719543, -517.8312988, 518.7620850

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099820, upper bound: 2585.4097337
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098128, upper bound: 2585.4097340
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -934.1843262, 1773.8087158, -940.8821411, 1786.6137695, -2720.7980957, 2714.6909180
1: -176.0939178, 229.1551514, -177.3397827, 230.8084869, -406.9023743, 406.4949341
2: -135.6017303, 306.0361633, -136.5652008, 308.2008057, -443.8025513, 442.6013489
3: -133.3491821, 397.2254028, -134.2920074, 400.0929260, -533.4421387, 531.5173340
4: -116.3877640, 384.8306885, -117.2141190, 387.5705872, -503.9582825, 502.0447998

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068009, upper bound: 2585.4094234
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096251
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -979.5281372, 1857.0981445, -940.8821411, 1786.6137695, -2766.1418457, 2797.9802246
1: -184.3682556, 239.9975433, -177.3397827, 230.8084869, -415.1767578, 417.3373413
2: -142.1612396, 319.2386780, -136.5652008, 308.2008057, -450.3620605, 455.8038635
3: -140.0028534, 415.8987122, -134.2920074, 400.0929260, -540.0957642, 550.1906128
4: -122.0578537, 401.4636536, -117.2141190, 387.5705872, -509.6284180, 518.6777954

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068014, upper bound: 2585.4094242
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096264
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -929.2895508, 1759.9956055, -896.5715332, 1715.8026123, -2645.0922852, 2656.5671387
1: -174.5885925, 227.5011444, -171.1359558, 221.3424377, -395.9310303, 398.6370850
2: -134.6994476, 304.2665100, -131.4639130, 293.0836792, -427.7831116, 435.7304077
3: -132.5598602, 394.7305298, -129.2513123, 382.9027405, -515.4625854, 523.9818115
4: -115.5710678, 382.5731506, -112.8719025, 368.2866211, -483.8576965, 495.4450073

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4078398, upper bound: 2585.4094680
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4078389, upper bound: 2585.4094694
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -968.0769043, 1855.0177002, -2809.3037109, 2779.7434082
1: -179.8823242, 234.0388947, -185.6766510, 239.1430511, -419.0253906, 419.7155457
2: -138.5360565, 312.7053833, -141.7303772, 317.3461304, -455.8821716, 454.4357300
3: -136.2803802, 405.7934875, -138.5716400, 413.3680725, -549.6483154, 544.3650513
4: -118.9099197, 393.1340637, -121.6171265, 399.0167236, -517.9266357, 514.7511597

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098010, upper bound: 2585.4068870
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098010, upper bound: 2585.4096305
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -947.2660522, 1797.3228760, -982.2663574, 1862.2283936, -2809.4943848, 2779.5893555
1: -178.4807892, 232.2029572, -184.8478394, 240.6604156, -419.1412048, 417.0507812
2: -137.4472046, 310.1622620, -142.5335999, 320.1076355, -457.5548096, 452.6958618
3: -135.2370911, 402.6688232, -140.3580475, 417.0577698, -552.2946777, 543.0268555
4: -117.9695969, 389.9200134, -122.3712540, 402.5979614, -520.5675659, 512.2912598

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067477, upper bound: 2585.4097113
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098316
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -989.3084106, 1873.9550781, -982.2663574, 1862.2283936, -2851.5368652, 2856.2211914
1: -186.0719604, 242.2192078, -184.8478394, 240.6604156, -426.7323608, 427.0669861
2: -143.4731293, 322.3971863, -142.5335999, 320.1076355, -463.5807495, 464.9307861
3: -141.3676910, 419.8358154, -140.3580475, 417.0577698, -558.4254761, 560.1937866
4: -123.1527328, 405.3808289, -122.3712540, 402.5979614, -525.7506714, 527.7519531

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067477, upper bound: 2585.4097006
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096243, upper bound: 2585.4098316
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -941.8275146, 1782.5416260, -928.9635010, 1772.3017578, -2714.1291504, 2711.5051270
1: -176.9403229, 230.4262085, -176.6190338, 228.8612518, -405.8015442, 407.0452271
2: -136.4766998, 308.0708618, -136.0166168, 302.1490479, -438.6257019, 444.0874634
3: -134.4544983, 399.9089661, -133.8398743, 395.7933655, -530.2478638, 533.7488403
4: -117.1299438, 387.2513733, -116.7292023, 379.6192932, -496.7492065, 503.9805298

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075223, upper bound: 2585.4096787
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4072569, upper bound: 2585.4095990
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4072377, upper bound: 2585.4095920
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -966.2313843, 1833.2004395, -1006.7078857, 1924.5274658, -2890.7587891, 2839.9077148
1: -182.0998230, 236.8301697, -192.4786072, 248.2474976, -430.3472900, 429.3087769
2: -140.2225494, 316.3553467, -147.2267303, 328.7904663, -469.0130005, 463.5820923
3: -138.0129700, 410.7770386, -144.1275482, 429.1018982, -567.1148682, 554.9045410
4: -120.3594055, 397.6459351, -126.3191147, 413.3293457, -533.6887207, 523.9650269

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096246, upper bound: 2585.4073355
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096246, upper bound: 2585.4098379
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -934.1843262, 1773.8087158, -986.2863159, 1869.9299316, -2804.1142578, 2760.0949707
1: -176.0939178, 229.1551514, -185.6130371, 241.6552277, -417.7490845, 414.7681885
2: -135.6017303, 306.0361633, -143.1280365, 321.4186707, -457.0203857, 449.1641846
3: -133.3491821, 397.2254028, -140.9549103, 418.7827454, -552.1318359, 538.1802979
4: -116.3877640, 384.8306885, -122.8881760, 404.2409363, -520.6286621, 507.7188721

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094155, upper bound: 2585.4067462
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096225, upper bound: 2585.4096226
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -979.5281372, 1857.0981445, -986.2863159, 1869.9299316, -2849.4580078, 2843.3842773
1: -184.3682556, 239.9975433, -185.6130371, 241.6552277, -426.0234680, 425.6105347
2: -142.1612396, 319.2386780, -143.1280365, 321.4186707, -463.5798950, 462.3666992
3: -140.0028534, 415.8987122, -140.9549103, 418.7827454, -558.7855225, 556.8536377
4: -122.0578537, 401.4636536, -122.8881760, 404.2409363, -526.2987061, 524.3518066

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067462, upper bound: 2585.4094170
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096225
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -929.2843018, 1759.9858398, -932.7388916, 1780.1199951, -2709.4042969, 2692.7246094
1: -174.5875854, 227.4998474, -177.3915253, 229.8259277, -404.4135132, 404.8913574
2: -134.6987305, 304.2648010, -136.5885620, 303.4174194, -438.1161499, 440.8533630
3: -132.5590973, 394.7283020, -134.3992004, 397.4619751, -530.0210571, 529.1274414
4: -115.5704269, 382.5710144, -117.2307968, 381.2058105, -496.7762451, 499.8018188

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068217, upper bound: 2585.4068549
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075201, upper bound: 2585.4094595
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -954.2865601, 1811.6665039, -1010.4092407, 1932.3514404, -2886.6379395, 2822.0756836
1: -179.8823242, 234.0388947, -193.2518768, 249.2224426, -429.1047668, 427.2907104
2: -138.5360565, 312.7053833, -147.8092957, 330.0435791, -468.5795898, 460.5146790
3: -136.2803802, 405.7934875, -144.6862030, 430.7590332, -567.0393677, 550.4795532
4: -118.9099197, 393.1340637, -126.8240433, 414.9025269, -533.8124390, 519.9580078

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094272, upper bound: 2585.4068872
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096215, upper bound: 2585.4096293
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -993.6475220, 1896.5800781, -937.1047363, 1779.4067383, -2773.0541992, 2833.6848145
1: -189.7981720, 244.7791901, -176.6248169, 229.8726501, -419.6708374, 421.4039917
2: -145.1436310, 324.8939209, -136.0071716, 306.9665222, -452.1101685, 460.9010620
3: -142.2358246, 423.5339661, -133.7311707, 398.4748535, -540.7106934, 557.2650757
4: -124.5412140, 408.3838196, -116.7290802, 386.0252991, -510.5665283, 525.1129150

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095095, upper bound: 2585.4075254
time: 0.76 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097276, upper bound: 2585.4097856
time: 0.74 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1029.8892822, 1963.0853271, -937.1047363, 1779.4067383, -2809.2958984, 2900.1894531
1: -196.2228851, 253.4745789, -176.6248169, 229.8726501, -426.0955200, 430.0993958
2: -150.3553009, 335.8014221, -136.0071716, 306.9665222, -457.3217163, 471.8085327
3: -147.4473724, 438.5150452, -133.7311707, 398.4748535, -545.9222412, 572.2462158
4: -128.9989624, 422.1042175, -116.7290802, 386.0252991, -515.0242310, 538.8331909

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095095, upper bound: 2585.4075237
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
time: 0.67 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -993.6475220, 1896.5800781, -967.5244141, 1852.9454346, -2846.5930176, 2864.1044922
1: -189.7981720, 244.7791901, -185.4743958, 238.9097748, -428.7079468, 430.2536011
2: -145.1436310, 324.8939209, -141.5873871, 317.0906677, -462.2343140, 466.4812927
3: -142.2358246, 423.5339661, -138.4504395, 412.9884644, -555.2241211, 561.9841919
4: -124.5412140, 408.3838196, -121.4871521, 398.6987000, -523.2398682, 529.8709717

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075557, upper bound: 2585.4096518
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099496, upper bound: 2585.4097856
time: 0.88 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1029.8892822, 1963.0853271, -967.5244141, 1852.9454346, -2882.8347168, 2930.6093750
1: -196.2228851, 253.4745789, -185.4743958, 238.9097748, -435.1326599, 438.9489746
2: -150.3553009, 335.8014221, -141.5873871, 317.0906677, -467.4458923, 477.3887634
3: -147.4473724, 438.5150452, -138.4504395, 412.9884644, -560.4356689, 576.9654541
4: -128.9989624, 422.1042175, -121.4871521, 398.6987000, -527.6975708, 543.5912476

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075558, upper bound: 2585.4096504
time: 0.79 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099496, upper bound: 2585.4097871
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -993.6475220, 1896.5800781, -982.2663574, 1862.2283936, -2855.8759766, 2878.8464355
1: -189.7981720, 244.7791901, -184.8478394, 240.6604156, -430.4585876, 429.6270142
2: -145.1436310, 324.8939209, -142.5335999, 320.1076355, -465.2512817, 467.4275208
3: -142.2358246, 423.5339661, -140.3580475, 417.0577698, -559.2935791, 563.8919678
4: -124.5412140, 408.3838196, -122.3712540, 402.5979614, -527.1391602, 530.7550659

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094694, upper bound: 2585.4078993
time: 0.68 seconds

## Relational analysis of NS_A2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094694, upper bound: 2585.4100898
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -993.6475220, 1896.5800781, -1009.1226196, 1928.9887695, -2922.6362305, 2905.7026367
1: -189.7981720, 244.7791901, -192.9198761, 248.8334808, -438.6316528, 437.6990662
2: -145.1436310, 324.8939209, -147.5729675, 329.5365295, -474.6801758, 472.4668884
3: -142.2358246, 423.5339661, -144.4653015, 430.1126709, -572.3485107, 567.9991455
4: -124.5412140, 408.3838196, -126.6131210, 414.2673340, -538.8085327, 534.9969482

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068900, upper bound: 2585.4099303
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096329, upper bound: 2585.4100940
time: 0.79 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1029.8892822, 1963.0853271, -982.2663574, 1862.2283936, -2892.1176758, 2945.3513184
1: -196.2228851, 253.4745789, -184.8478394, 240.6604156, -436.8833008, 438.3224182
2: -150.3553009, 335.8014221, -142.5335999, 320.1076355, -470.4628601, 478.3350220
3: -147.4473724, 438.5150452, -140.3580475, 417.0577698, -564.5051270, 578.8731079
4: -128.9989624, 422.1042175, -122.3712540, 402.5979614, -531.5969238, 544.4754028

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094537, upper bound: 2585.4075222
time: 0.71 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096280, upper bound: 2585.4097831
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1029.8892822, 1963.0853271, -1009.1226196, 1928.9887695, -2958.8779297, 2972.2075195
1: -196.2228851, 253.4745789, -192.9198761, 248.8334808, -445.0563660, 446.3944702
2: -150.3553009, 335.8014221, -147.5729675, 329.5365295, -479.8917236, 483.3743896
3: -147.4473724, 438.5150452, -144.4653015, 430.1126709, -577.5600586, 582.9803467
4: -128.9989624, 422.1042175, -126.6131210, 414.2673340, -543.2662964, 548.7173462

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068612, upper bound: 2585.4096467
time: 0.75 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097832
time: 0.72 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -964.6633911, 1849.4851074, -940.8821411, 1786.6137695, -2751.2770996, 2790.3671875
1: -185.1419525, 238.3877411, -177.3397827, 230.8084869, -415.9504395, 415.7275391
2: -141.2984924, 316.2402954, -136.5652008, 308.2008057, -449.4992676, 452.8054810
3: -138.0811615, 412.0292664, -134.2920074, 400.0929260, -538.1740723, 546.3211670
4: -121.2370834, 397.6160278, -117.2141190, 387.5705872, -508.8076782, 514.8300781

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095144, upper bound: 2585.4076442
time: 0.74 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097286, upper bound: 2585.4097600
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1006.0653076, 1925.2583008, -940.8821411, 1786.6137695, -2792.6789551, 2866.1403809
1: -192.5580902, 248.2636566, -177.3397827, 230.8084869, -423.3665771, 425.6034241
2: -147.2548065, 328.6759949, -136.5652008, 308.2008057, -455.4556274, 465.2411804
3: -144.0786743, 429.0533142, -134.2920074, 400.0929260, -544.1715698, 563.3452759
4: -126.3390961, 413.1736755, -117.2141190, 387.5705872, -513.9095459, 530.3878174

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095126, upper bound: 2585.4076442
time: 0.75 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097286, upper bound: 2585.4097599
time: 0.67 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -964.6633911, 1849.4851074, -971.1600342, 1860.6889648, -2825.3522949, 2820.6450195
1: -185.1419525, 238.3877411, -186.2384338, 239.8831482, -425.0250854, 424.6261597
2: -141.2984924, 316.2402954, -142.1675415, 318.3192749, -459.6177673, 458.4078064
3: -138.0811615, 412.0292664, -138.9995880, 414.6497803, -552.7309570, 551.0288696
4: -121.2370834, 397.6160278, -121.9870453, 400.2417603, -521.4787598, 519.6030273

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096527, upper bound: 2585.4082352
time: 0.84 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099145, upper bound: 2585.4096736
time: 0.90 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4099377, upper bound: 2585.4096724
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1006.0653076, 1925.2583008, -971.1600342, 1860.6889648, -2866.7541504, 2896.4184570
1: -192.5580902, 248.2636566, -186.2384338, 239.8831482, -432.4412231, 434.5020447
2: -147.2548065, 328.6759949, -142.1675415, 318.3192749, -465.5740967, 470.8435059
3: -144.0786743, 429.0533142, -138.9995880, 414.6497803, -558.7284546, 568.0529175
4: -126.3390961, 413.1736755, -121.9870453, 400.2417603, -526.5808105, 535.1607056

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096527, upper bound: 2585.4082352
time: 0.78 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068014, upper bound: 2585.4096519
time: 0.75 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100025, upper bound: 2585.4097634
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -891.3884888, 1707.3693848, -1001.9992065, 1913.4638672, -2804.8522949, 2709.3686523
1: -171.1886902, 220.3511658, -191.3515320, 247.0107117, -418.1993408, 411.7026367
2: -130.6422577, 292.6513977, -146.4810028, 327.4034729, -458.0457153, 439.1323853
3: -127.8786011, 380.7952881, -143.4404755, 427.0317383, -554.9102783, 524.2357178
4: -112.1542130, 367.7854919, -125.6613464, 411.5628662, -523.7171021, 493.4467773

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4091920, upper bound: 2585.4096150
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4091930, upper bound: 2585.4096150
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -937.8948975, 1802.9072266, -1012.2212524, 1936.7530518, -2874.6474609, 2815.1284180
1: -180.5133514, 232.2747192, -193.6728668, 249.7654724, -430.2788086, 425.9475708
2: -137.6602783, 307.8772888, -148.1271667, 330.9413757, -468.6016541, 456.0044556
3: -134.4234467, 401.3728027, -144.9312744, 431.7055969, -566.1290283, 546.3039551
4: -118.1465836, 386.8541260, -127.0842133, 415.9862061, -534.1328125, 513.9382935

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4091930, upper bound: 2585.4096150
time: 0.74 seconds

## Relational analysis of NS_A2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4091931, upper bound: 2585.4096135
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1006.0653076, 1925.2583008, -986.2863159, 1869.9299316, -2875.9951172, 2911.5437012
1: -192.5580902, 248.2636566, -185.6130371, 241.6552277, -434.2132874, 433.8766174
2: -147.2548065, 328.6759949, -143.1280365, 321.4186707, -468.6734619, 471.8040161
3: -144.0786743, 429.0533142, -140.9549103, 418.7827454, -562.8612671, 570.0082397
4: -126.3390961, 413.1736755, -122.8881760, 404.2409363, -530.5799561, 536.0618286

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094594, upper bound: 2585.4075721
time: 0.69 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096294, upper bound: 2585.4097561
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1006.0653076, 1925.2583008, -1012.8219604, 1936.8083496, -2942.8735352, 2938.0798340
1: -192.5580902, 248.2636566, -193.6926422, 249.8078156, -442.3659058, 441.9562378
2: -147.2548065, 328.6759949, -148.1551361, 330.7892761, -478.0440674, 476.8311157
3: -144.0786743, 429.0533142, -145.0234528, 431.7693176, -575.8480225, 574.0767822
4: -126.3390961, 413.1736755, -127.1176224, 415.8401184, -542.1791992, 540.2913208

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095754, upper bound: 2585.4096703
time: 0.72 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068815, upper bound: 2585.4096476
time: 0.75 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096295, upper bound: 2585.4097615
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.54 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068025, upper bound: 2585.4097088
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098354
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068029, upper bound: 2585.4097088
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097243, upper bound: 2585.4098358
NS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4073584, upper bound: 2585.4096037
NS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4075149, upper bound: 2585.4096120
NS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4099820, upper bound: 2585.4097337
NS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4098128, upper bound: 2585.4097340
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068009, upper bound: 2585.4094234
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096251
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068014, upper bound: 2585.4094242
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097228, upper bound: 2585.4096264
NS_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4078398, upper bound: 2585.4094680
NS_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4078389, upper bound: 2585.4094694
NS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4098010, upper bound: 2585.4068870
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4098010, upper bound: 2585.4096305
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4067477, upper bound: 2585.4097113
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096248, upper bound: 2585.4098316
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4067477, upper bound: 2585.4097006
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096243, upper bound: 2585.4098316
NS_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4072569, upper bound: 2585.4095990
NS_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4072377, upper bound: 2585.4095920
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096246, upper bound: 2585.4073355
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096246, upper bound: 2585.4098379
NS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4094155, upper bound: 2585.4067462
NS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096225, upper bound: 2585.4096226
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4067462, upper bound: 2585.4094170
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096233, upper bound: 2585.4096225
NS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068217, upper bound: 2585.4068549
NS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4075201, upper bound: 2585.4094595
NS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4094272, upper bound: 2585.4068872
NS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096215, upper bound: 2585.4096293
NS_A2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4095095, upper bound: 2585.4075254
NS_A2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097276, upper bound: 2585.4097856
NS_A2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4095095, upper bound: 2585.4075237
NS_A2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097284, upper bound: 2585.4097856
NS_A2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4075557, upper bound: 2585.4096518
NS_A2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4099496, upper bound: 2585.4097856
NS_A2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4075558, upper bound: 2585.4096504
NS_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4099496, upper bound: 2585.4097871
NS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4094694, upper bound: 2585.4078993
NS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4094694, upper bound: 2585.4100898
NS_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068900, upper bound: 2585.4099303
NS_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096329, upper bound: 2585.4100940
NS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4094537, upper bound: 2585.4075222
NS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096280, upper bound: 2585.4097831
NS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068612, upper bound: 2585.4096467
NS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096288, upper bound: 2585.4097832
NS_A2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4095144, upper bound: 2585.4076442
NS_A2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097286, upper bound: 2585.4097600
NS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4095126, upper bound: 2585.4076442
NS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4097286, upper bound: 2585.4097599
NS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4099145, upper bound: 2585.4096736
NS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4099377, upper bound: 2585.4096724
NS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068014, upper bound: 2585.4096519
NS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4100025, upper bound: 2585.4097634
NS_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4091920, upper bound: 2585.4096150
NS_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4091930, upper bound: 2585.4096150
NS_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4091930, upper bound: 2585.4096150
NS_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4091931, upper bound: 2585.4096135
NS_A2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4094594, upper bound: 2585.4075721
NS_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096294, upper bound: 2585.4097561
NS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4068815, upper bound: 2585.4096476
NS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.54
Output dim: 0, lower bound: -2585.4096295, upper bound: 2585.4097615

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -922.6881104, 1746.2954102, -865.4262695, 1637.9210205, -2560.6091309, 2611.7211914
1: -173.2716370, 225.7476959, -161.8852386, 211.9340820, -385.2057190, 387.6329346
2: -133.6712036, 301.8273926, -125.6711655, 282.1606750, -415.8318787, 427.4985657
3: -131.6461945, 391.7276306, -124.3798141, 367.4904785, -499.1366577, 516.1074219
4: -114.7126770, 379.4643555, -107.9461899, 354.7430115, -469.4556580, 487.4105530

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068083, upper bound: 2585.4099459
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068083, upper bound: 2585.4099483
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -947.2660522, 1797.3228760, -934.0051880, 1773.6594238, -2720.9250488, 2731.3281250
1: -178.4807892, 232.2029572, -176.0584259, 229.1229858, -407.6037598, 408.2613831
2: -137.4472046, 310.1622620, -135.5645752, 305.9961243, -443.4432983, 445.7267761
3: -135.2370911, 402.6688232, -133.3003387, 397.1767883, -532.4136963, 535.9691162
4: -117.9695969, 389.9200134, -116.3544617, 384.7982788, -502.7678528, 506.2744751

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094957, upper bound: 2585.4077576
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094957, upper bound: 2585.4100902
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -960.6630249, 1815.7409668, -865.4262695, 1637.9210205, -2598.5839844, 2681.1667480
1: -180.1434326, 234.8517151, -161.8852386, 211.9340820, -392.0775146, 396.7369385
2: -139.1371002, 312.8480225, -125.6711655, 282.1606750, -421.2977600, 438.5191956
3: -137.1456299, 407.2671204, -124.3798141, 367.4904785, -504.6360779, 531.6469116
4: -119.3864136, 393.3757019, -107.9461899, 354.7430115, -474.1293945, 501.3218994

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068029, upper bound: 2585.4097088
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068029, upper bound: 2585.4097065
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -989.3084106, 1873.9550781, -934.0051880, 1773.6594238, -2762.9677734, 2807.9602051
1: -186.0719604, 242.2192078, -176.0584259, 229.1229858, -415.1949463, 418.2776489
2: -143.4731293, 322.3971863, -135.5645752, 305.9961243, -449.4692383, 457.9617004
3: -141.3676910, 419.8358154, -133.3003387, 397.1767883, -538.5444336, 553.1359863
4: -123.1527328, 405.3808289, -116.3544617, 384.7982788, -507.9510193, 521.7352295

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094792, upper bound: 2585.4073333
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094792, upper bound: 2585.4098356
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -939.2026367, 1777.0061035, -838.7870483, 1589.9780273, -2529.1806641, 2615.7932129
1: -176.3461609, 229.7276459, -158.3023987, 205.7506561, -382.0968018, 388.0300293
2: -136.0600891, 307.1770935, -122.0985489, 272.2884521, -408.3485413, 429.2756348
3: -134.0643768, 398.7312317, -120.2046280, 356.3244934, -490.3888550, 518.9357910
4: -116.7750549, 386.1307983, -104.8040848, 342.5299072, -459.3049622, 490.9348755

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4073575, upper bound: 2585.4096037
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4073584, upper bound: 2585.4096037
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -941.7072144, 1782.3037109, -881.1347046, 1683.8748779, -2625.5817871, 2663.4384766
1: -176.9173126, 230.3961639, -167.9110260, 217.3264771, -394.2437744, 398.3071594
2: -136.4590302, 308.0299683, -129.0876160, 287.8518066, -424.3108521, 437.1174927
3: -134.4370117, 399.8563843, -126.9876099, 376.0663147, -510.5033264, 526.8439941
4: -117.1146545, 387.2001953, -110.8140411, 361.7500916, -478.8647156, 498.0142212

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075150, upper bound: 2585.4096120
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075150, upper bound: 2585.4096099
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -955.6646118, 1814.1085205, -911.6854248, 1753.1096191, -2708.7734375, 2725.7934570
1: -180.2641754, 234.3647614, -175.7881165, 225.8921661, -406.1563416, 410.1528931
2: -138.7701416, 312.8388672, -133.9313965, 298.7709656, -437.5410767, 446.7702637
3: -136.6250763, 406.3507690, -131.0888519, 389.9076843, -526.5327148, 537.4396362
4: -119.1316986, 393.2033081, -115.0064163, 375.5266418, -494.6583252, 508.2097168

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4092888, upper bound: 2585.4095571
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098800, upper bound: 2585.4095970
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -960.3314819, 1821.8298340, -1452.5034180, 2821.3835449, -3781.7148438, 3274.3330078
1: -180.9969330, 235.3681793, -282.5899353, 363.5823059, -544.5792236, 517.7298584
2: -139.3585358, 314.3359375, -215.6952515, 467.3145447, -606.6730957, 530.0311279
3: -137.1802521, 408.2200623, -214.0176849, 622.5286865, -759.7089233, 622.2376099
4: -119.6216812, 395.1138916, -185.6301270, 587.0408325, -706.6624146, 580.7440186

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098123, upper bound: 2585.4097319
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098128, upper bound: 2585.4097340
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -909.0422363, 1721.7619629, -869.3954468, 1645.4062500, -2554.4484863, 2591.1574707
1: -170.7460327, 222.5703888, -162.6181183, 212.9002228, -383.6462097, 385.1885071
2: -131.7330627, 297.5618591, -126.2459030, 283.4497681, -415.1828308, 423.8077698
3: -129.5926056, 386.0936279, -124.9638748, 369.1814880, -498.7740784, 511.0574646
4: -113.0179596, 374.2208252, -108.4446945, 356.3602600, -469.3782043, 482.6655273

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067387, upper bound: 2585.4067387
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067387, upper bound: 2585.4094730
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -934.1843262, 1773.8087158, -937.7937012, 1780.8863525, -2715.0703125, 2711.6025391
1: -176.0939178, 229.1551514, -176.7753754, 230.0615387, -406.1554260, 405.9305420
2: -135.6017303, 306.0361633, -136.1241302, 307.2338867, -442.8356323, 442.1602783
3: -133.3491821, 397.2254028, -133.8626862, 398.7993774, -532.1484375, 531.0880127
4: -116.3877640, 384.8306885, -116.8407440, 386.3480835, -502.7358093, 501.6714172

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094730, upper bound: 2585.4068045
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094730, upper bound: 2585.4097249
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -951.5385742, 1799.9569092, -869.3954468, 1645.4062500, -2596.9445801, 2669.3520508
1: -178.5182648, 232.7639618, -162.6181183, 212.9002228, -391.4184570, 395.3820801
2: -137.8946686, 309.8800964, -126.2459030, 283.4497681, -421.3444214, 436.1260071
3: -135.8476715, 403.6037903, -124.9638748, 369.1814880, -505.0291138, 528.5676880
4: -118.3435669, 389.7208252, -108.4446945, 356.3602600, -474.7037964, 498.1655273

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067418, upper bound: 2585.4068184
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067418, upper bound: 2585.4094218
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -979.5281372, 1857.0981445, -937.7937012, 1780.8863525, -2760.4140625, 2794.8918457
1: -184.3682556, 239.9975433, -176.7753754, 230.0615387, -414.4298096, 416.7729187
2: -142.1612396, 319.2386780, -136.1241302, 307.2338867, -449.3951111, 455.3627930
3: -140.0028534, 415.8987122, -133.8626862, 398.7993774, -538.8021240, 549.7613525
4: -122.0578537, 401.4636536, -116.8407440, 386.3480835, -508.4059448, 518.3043823

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094761, upper bound: 2585.4068826
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094758, upper bound: 2585.4096266
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -929.2869873, 1759.9910889, -923.9686279, 1761.5732422, -2690.8603516, 2683.9597168
1: -174.5880890, 227.5004883, -175.7155151, 227.5164642, -402.1045532, 403.2160034
2: -134.6990967, 304.2656860, -135.2337189, 301.0017395, -435.7008362, 439.4993896
3: -132.5595093, 394.7294312, -133.2341156, 393.9005737, -526.4600830, 527.9635010
4: -115.5707550, 382.5721130, -116.1150436, 378.0999146, -493.6706543, 498.6871643

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071344, upper bound: 2585.4092662
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4060957, upper bound: 2585.4092129
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -929.2856445, 1759.9880371, -890.4637451, 1705.3153076, -2634.6005859, 2650.4516602
1: -174.5878143, 227.5001373, -170.0992279, 219.9367981, -394.5245972, 397.5993652
2: -134.6988831, 304.2652283, -130.6440735, 291.1494751, -425.8483582, 434.9091797
3: -132.5592804, 394.7287598, -128.3895721, 380.3972778, -512.9564819, 523.1183472
4: -115.5705795, 382.5714722, -112.1598663, 365.8447266, -481.4153137, 494.7313232

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071344, upper bound: 2585.4092644
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068904, upper bound: 2585.4092533
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -882.4190063, 1669.7542725, -966.3146362, 1851.4411621, -2733.8598633, 2636.0688477
1: -165.0582275, 216.0425262, -185.3203125, 238.6862640, -403.7445068, 401.3628235
2: -128.1609955, 287.7140503, -141.4640808, 316.7324219, -444.8934326, 429.1781311
3: -126.9001312, 374.7043457, -138.3081665, 412.5875244, -539.4876709, 513.0124512
4: -110.0915451, 361.6379700, -121.3874283, 398.2480164, -508.3395691, 483.0253296

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4077871, upper bound: 2585.4068869
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4077874, upper bound: 2585.4068894
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -951.1141968, 1805.7846680, -968.0769043, 1855.0177002, -2806.1315918, 2773.8615723
1: -179.3026276, 233.2731323, -185.6766510, 239.1430511, -418.4456787, 418.9497681
2: -138.0829620, 311.7184143, -141.7303772, 317.3461304, -455.4290771, 453.4487610
3: -135.8392029, 404.4661255, -138.5716400, 413.3680725, -549.2070312, 543.0377808
4: -118.5261917, 391.8865967, -121.6171265, 399.0167236, -517.5429077, 513.5037231

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4077874, upper bound: 2585.4096305
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4067419, upper bound: 2585.4096305
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -922.6803589, 1746.2799072, -904.5458374, 1707.6928711, -2630.3732910, 2650.8256836
1: -173.2700806, 225.7457123, -168.7242889, 221.0678558, -394.3379517, 394.4699707
2: -133.6700745, 301.8247681, -131.2135773, 292.8759460, -426.5460205, 433.0383301
3: -131.6450500, 391.7242432, -130.0078430, 383.2550659, -514.9001465, 521.7320557
4: -114.7116852, 379.4610291, -112.6938858, 368.2790527, -482.9906921, 492.1549072

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068403, upper bound: 2585.4076935
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068403, upper bound: 2585.4099507
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -947.2660522, 1797.3228760, -979.8082275, 1857.6679688, -2804.9340820, 2777.1311035
1: -178.4807892, 232.2029572, -184.3983765, 240.0606995, -418.5414429, 416.6013184
2: -137.4472046, 310.1622620, -142.1808167, 319.3466797, -456.7938538, 452.3430786
3: -135.2370911, 402.6688232, -140.0132141, 416.0306091, -551.2674561, 542.6819458
4: -117.9695969, 389.9200134, -122.0713348, 401.6403198, -519.6098022, 511.9913330

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094469, upper bound: 2585.4077533
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094469, upper bound: 2585.4100863
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -960.6553955, 1815.7259521, -904.5458374, 1707.6928711, -2668.3481445, 2720.2717285
1: -180.1419220, 234.8497925, -168.7242889, 221.0678558, -401.2097778, 403.5740051
2: -139.1359558, 312.8454590, -131.2135773, 292.8759460, -432.0119019, 444.0590210
3: -137.1445160, 407.2638245, -130.0078430, 383.2550659, -520.3994751, 537.2716064
4: -119.3854446, 393.3724670, -112.6938858, 368.2790527, -487.6644287, 506.0663452

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068216, upper bound: 2585.4072597
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068217, upper bound: 2585.4097009
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -989.3084106, 1873.9550781, -979.8082275, 1857.6679688, -2846.9763184, 2853.7631836
1: -186.0719604, 242.2192078, -184.3983765, 240.0606995, -426.1326599, 426.6175232
2: -143.4731293, 322.3971863, -142.1808167, 319.3466797, -462.8198242, 464.5780029
3: -141.3676910, 419.8358154, -140.0132141, 416.0306091, -557.3981934, 559.8488159
4: -123.1527328, 405.3808289, -122.0713348, 401.6403198, -524.7930298, 527.4521484

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094304, upper bound: 2585.4073297
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094304, upper bound: 2585.4098318
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -939.1975708, 1776.9959717, -875.8986206, 1656.3443604, -2595.5417480, 2652.8945312
1: -176.3451233, 229.7263794, -164.7714996, 214.5046234, -390.8497314, 394.4978333
2: -136.0593262, 307.1753845, -127.3893433, 283.4508972, -419.5102234, 434.5647278
3: -134.0636139, 398.7290649, -125.5049515, 371.4254761, -505.4890747, 524.2339478
4: -116.7743912, 386.1286316, -109.2841339, 356.4649963, -473.2393494, 495.4127808

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071231, upper bound: 2585.4095905
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071231, upper bound: 2585.4095969
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -941.7019653, 1782.2935791, -916.5977783, 1746.8737793, -2688.5756836, 2698.8908691
1: -176.9162750, 230.3948212, -174.0167694, 225.6623993, -402.5786438, 404.4115906
2: -136.4582672, 308.0281372, -134.1143646, 297.9292297, -434.3874817, 442.1425171
3: -134.4362640, 399.8541565, -132.0320740, 390.3438416, -524.7800903, 531.8862305
4: -117.1139908, 387.1980591, -115.0902023, 374.3935242, -491.5074768, 502.2882690

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4072368, upper bound: 2585.4095935
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4072377, upper bound: 2585.4095920
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -899.7810669, 1702.0668945, -1004.8545532, 1920.7454834, -2820.5266113, 2706.9213867
1: -168.4978790, 220.3036499, -192.1010437, 247.7658081, -416.2636719, 412.4046936
2: -130.7496185, 292.9645996, -146.9459229, 328.1425171, -458.8921509, 439.9105225
3: -129.5351715, 382.0115051, -143.8493500, 428.2790222, -557.8140259, 525.8608398
4: -112.3325195, 368.1411133, -126.0769424, 412.5183411, -524.8507690, 494.2180481

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068249, upper bound: 2585.4073355
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4074655, upper bound: 2585.4073355
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -963.0001831, 1827.2402344, -1006.7078857, 1924.5274658, -2887.5275879, 2833.9482422
1: -181.5138550, 236.0523071, -192.4786072, 248.2474976, -429.7613525, 428.5309143
2: -139.7628326, 315.3538208, -147.2267303, 328.7904663, -468.5532837, 462.5805664
3: -137.5651245, 409.4308777, -144.1275482, 429.1018982, -566.6669922, 553.5582886
4: -119.9693985, 396.3820496, -126.3191147, 413.3293457, -533.2986450, 522.7011108

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075353, upper bound: 2585.4098381
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4075353, upper bound: 2585.4098379
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -863.0587158, 1633.2004395, -958.2699585, 1812.7895508, -2675.8481445, 2591.4704590
1: -161.4168396, 211.3195038, -179.7627411, 234.4207916, -395.8376465, 391.0822144
2: -125.3243942, 281.3925476, -138.8613892, 312.0537720, -437.3781738, 420.2539368
3: -124.0609360, 366.4711914, -136.7984009, 406.4805298, -530.5413818, 503.2695923
4: -107.6506195, 353.7560730, -119.1739731, 392.4892273, -500.1398315, 472.9300537

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068184, upper bound: 2585.4067411
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068184, upper bound: 2585.4068009
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -931.1179199, 1768.1240234, -986.2863159, 1869.9299316, -2801.0478516, 2754.4094238
1: -175.5334778, 228.4137421, -185.6130371, 241.6552277, -417.1886597, 414.0267639
2: -135.1639557, 305.0769043, -143.1280365, 321.4186707, -456.5826416, 448.2049561
3: -132.9230347, 395.9414978, -140.9549103, 418.7827454, -551.7056885, 536.8964233
4: -116.0171738, 383.6178894, -122.8881760, 404.2409363, -520.2580566, 506.5060730

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068844, upper bound: 2585.4094755
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4068844, upper bound: 2585.4097222
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -951.5305176, 1799.9409180, -908.6812744, 1715.5111084, -2667.0415039, 2708.6220703
1: -178.5166931, 232.7619324, -169.4923706, 222.0757446, -400.5924072, 402.2542419
2: -137.8934937, 309.8773804, -131.8128510, 294.2188110, -432.1123047, 441.6902466
3: -135.8464508, 403.6003113, -130.6141815, 385.0160828, -520.8625488, 534.2144775
4: -118.3425446, 389.7174377, -113.2132797, 369.9507446, -488.2932739, 502.9306946

Time for backsubstitution: 2.44 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.27 + 417.83 = 422.11 seconds
