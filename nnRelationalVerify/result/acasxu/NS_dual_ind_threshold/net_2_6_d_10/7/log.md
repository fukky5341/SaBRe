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
execution time: IAR + RelationalAnalysis = 2.49 + 1.94 = 4.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2585.4102985, upper bound: 2585.4102985

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101177, upper bound: 2585.4097073
time: 0.71 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4102985, upper bound: 2585.4102985
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.54 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 0, lower bound: -2585.4101177, upper bound: 2585.4097073
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 0, lower bound: -2585.4102985, upper bound: 2585.4102985

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -946.2250366, 1820.4763184, -975.8432007, 1873.0646973, -2819.2897949, 2796.3195801
1: -182.3161316, 234.3336182, -187.4984283, 241.3084106, -423.6245422, 421.8320312
2: -139.0035858, 310.7358398, -143.0946655, 320.1156921, -459.1192322, 453.8304749
3: -135.8534088, 405.3131409, -139.8996887, 417.3058777, -553.1593018, 545.2127075
4: -119.3248978, 390.3693542, -122.7939911, 402.4103088, -521.7351685, 513.1632690

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100812, upper bound: 2585.4096296
time: 0.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100250, upper bound: 2585.4096291
time: 0.69 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1081.1013184, 2067.1508789, -982.8639526, 1879.7427979, -2960.8437500, 3050.0148926
1: -206.9309540, 266.9175415, -188.1447754, 242.4290924, -449.3600159, 455.0622864
2: -158.5034332, 353.7703857, -143.7119751, 322.2461853, -480.7496338, 497.4822998
3: -156.5552826, 461.6896973, -140.6906586, 419.3562012, -575.9114380, 602.3803101
4: -136.3269501, 444.5697327, -123.3177643, 405.0934143, -541.4202881, 567.8875122

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4102899, upper bound: 2585.4101852
time: 0.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101846, upper bound: 2585.4101845
time: 0.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.81 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -2585.4100812, upper bound: 2585.4096296
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -2585.4100250, upper bound: 2585.4096291
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -2585.4102899, upper bound: 2585.4101852
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -2585.4101846, upper bound: 2585.4101845

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -935.9221191, 1801.9898682, -923.5573730, 1779.5109863, -2715.4331055, 2725.5468750
1: -180.5347290, 231.9355164, -178.4485474, 229.1216125, -409.6563416, 410.3840637
2: -137.5980377, 307.3533325, -135.9379730, 302.8637390, -440.4617920, 443.2912903
3: -134.5187378, 401.0539551, -133.0407257, 395.7307129, -530.2494507, 534.0946655
4: -118.1421280, 386.0942078, -116.7467651, 380.5956726, -498.7377930, 502.8409729

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100250, upper bound: 2585.4096272
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100250, upper bound: 2585.4096292
time: 0.78 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -941.0085449, 1810.2178955, -1468.1115723, 2855.4511719, -3796.4597168, 3278.3293457
1: -181.3125763, 233.0083160, -286.1052856, 367.7387085, -549.0512695, 519.1135864
2: -138.2225647, 308.9161682, -218.2546082, 472.6959229, -610.9182129, 527.1707764
3: -135.0941467, 403.0345764, -216.5485992, 629.8368530, -764.9308472, 619.5831909
4: -118.6521683, 388.1200562, -187.8576660, 593.7864990, -712.4386597, 575.9777222

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4095860
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098157, upper bound: 2585.4096249
time: 0.76 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1070.8181152, 2048.3952637, -930.3399048, 1785.6297607, -2856.4477539, 2978.7348633
1: -205.1308899, 264.4997864, -179.0679169, 230.1698303, -435.3007202, 443.5676575
2: -157.0847626, 350.3112793, -136.5225372, 304.8922119, -461.9769592, 486.8338013
3: -155.2310333, 457.3923035, -133.7936859, 397.5257874, -552.7567139, 591.1859131
4: -135.1315765, 440.1864014, -117.2392731, 383.1651001, -518.2966919, 557.4255371

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101846, upper bound: 2585.4101832
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101846, upper bound: 2585.4101846
time: 0.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1075.5102539, 2056.2878418, -1473.3000488, 2861.6699219, -3937.1799316, 3529.5878906
1: -205.8669128, 265.5101318, -286.6876526, 368.6657410, -574.5326538, 552.1977539
2: -157.6716614, 351.8943481, -218.7751160, 474.2380676, -631.9096680, 570.6694336
3: -155.7442017, 459.2568054, -217.1435242, 631.5089722, -787.2531738, 676.4003296
4: -135.6137238, 442.2301025, -188.3055267, 595.8483276, -731.4620361, 630.5354614

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098232, upper bound: 2585.4101013
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.89 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4100250, upper bound: 2585.4096272
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4100250, upper bound: 2585.4096292
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4095860
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4098157, upper bound: 2585.4096249
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4101846, upper bound: 2585.4101832
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4101846, upper bound: 2585.4101846
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4098232, upper bound: 2585.4101013
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -894.2085571, 1727.0761719, -923.5573730, 1779.5109863, -2673.7192383, 2650.6330566
1: -173.2766113, 222.1927490, -178.4485474, 229.1216125, -402.3981934, 400.6412964
2: -131.8608398, 293.6429443, -135.9379730, 302.8637390, -434.7245789, 429.5809326
3: -129.0511017, 383.8222656, -133.0407257, 395.7307129, -524.7817383, 516.8629761
4: -113.2994995, 368.8216858, -116.7467651, 380.5956726, -493.8951721, 485.5684509

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100740, upper bound: 2585.4096270
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098495, upper bound: 2585.4096252
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1440.9765625, 2806.4606934, -923.5573730, 1779.5109863, -3220.4873047, 3730.0180664
1: -281.2684326, 361.2047729, -178.4485474, 229.1216125, -510.3900452, 539.6533203
2: -214.4477234, 463.5421143, -135.9379730, 302.8637390, -517.3114014, 599.4799805
3: -212.7827301, 618.6784668, -133.0407257, 395.7307129, -608.5134277, 751.7191772
4: -184.6203003, 582.2339478, -116.7467651, 380.5956726, -565.2159424, 698.9806519

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100740, upper bound: 2585.4096271
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098495, upper bound: 2585.4096252
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -875.9515381, 1683.9498291, -1452.5133057, 2823.6838379, -3699.6352539, 3136.4626465
1: -168.9257812, 217.0014648, -282.9145508, 363.7425842, -532.6683350, 499.9160156
2: -128.7904816, 287.7196960, -215.9028625, 467.4870605, -596.2775269, 503.6225586
3: -126.0938187, 375.1668396, -214.2681732, 622.9960938, -749.0899048, 589.4348145
4: -110.6184235, 361.5398560, -185.8498993, 587.2172852, -697.8356934, 547.3897095

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097857, upper bound: 2585.4091981
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096161, upper bound: 2585.4091950
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -916.3449707, 1766.3250732, -1461.2872314, 2844.0925293, -3760.4375000, 3227.6123047
1: -176.9187164, 227.2515717, -284.9953308, 366.1720276, -543.0907593, 512.2468872
2: -134.8099976, 301.4415894, -217.3107300, 470.6563110, -605.4662476, 518.7523193
3: -131.8046265, 393.1307373, -215.5325165, 627.1069336, -758.9115601, 608.6630249
4: -115.7513123, 378.7052002, -187.0393066, 591.1686401, -706.9199219, 565.7445068

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096335, upper bound: 2585.4095931
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1029.7167969, 1973.6641846, -930.3399048, 1785.6297607, -2815.3461914, 2904.0039062
1: -197.8762207, 254.7213898, -179.0679169, 230.1698303, -428.0459900, 433.7892761
2: -151.3498230, 336.4825745, -136.5225372, 304.8922119, -456.2420044, 473.0050964
3: -149.8740692, 440.1276245, -133.7936859, 397.5257874, -547.3998413, 573.9212646
4: -130.2928619, 422.6629333, -117.2392731, 383.1651001, -513.4579468, 539.9021606

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101743, upper bound: 2585.4098236
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098216
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1583.8483887, 3061.5849609, -930.3399048, 1785.6297607, -3369.4780273, 3991.9248047
1: -306.8649597, 394.6599121, -179.0679169, 230.1698303, -537.0347900, 573.7278442
2: -234.5288239, 508.1390686, -136.5225372, 304.8922119, -539.4210205, 644.6614990
3: -234.4149933, 677.2192383, -133.7936859, 397.5257874, -631.9406128, 811.0128784
4: -202.1772308, 638.0159912, -117.2392731, 383.1651001, -585.3423462, 755.2551880

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4101743, upper bound: 2585.4098218
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098217
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -991.7341919, 1898.5687256, -1457.7967529, 2830.4384766, -3822.1726074, 3356.3654785
1: -190.3254242, 245.1598511, -283.5219116, 364.7237549, -555.0491333, 528.6817017
2: -145.5472260, 325.9342041, -216.4418182, 469.1009216, -614.6481323, 542.3760376
3: -143.5331879, 423.8326721, -214.8721313, 624.7386475, -768.2717285, 638.7047729
4: -125.1787186, 409.4576111, -186.3111267, 589.3631592, -714.5418091, 595.7687378

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096163, upper bound: 2585.4096275
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1058.2601318, 2023.6773682, -1466.2814941, 2849.9750977, -3908.2353516, 3489.9589844
1: -202.6812744, 261.3731384, -285.5537415, 367.0603027, -569.7414551, 546.9268799
2: -155.2660217, 345.3315125, -217.8088989, 472.1653137, -627.4312744, 563.1403809
3: -153.4738464, 451.8710022, -216.1063385, 628.6688232, -782.1425171, 667.9773560
4: -133.5735931, 433.8233948, -187.4615936, 593.1889648, -726.7625732, 621.2849121

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098198
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.88 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4100740, upper bound: 2585.4096270
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098495, upper bound: 2585.4096252
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4100740, upper bound: 2585.4096271
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098495, upper bound: 2585.4096252
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4097857, upper bound: 2585.4091981
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096161, upper bound: 2585.4091950
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096335, upper bound: 2585.4095931
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4101743, upper bound: 2585.4098236
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098216
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4101743, upper bound: 2585.4098218
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098217
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4096163, upper bound: 2585.4096275
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098198
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -875.9097290, 1691.1979980, -854.7002563, 1646.8237305, -2522.7333984, 2545.8981934
1: -169.7463684, 217.5869446, -165.3543701, 212.1933594, -381.9397278, 382.9413147
2: -129.1519928, 287.5867920, -125.9512558, 280.8149719, -409.9669800, 413.5380554
3: -126.3336639, 375.8075256, -123.5260849, 366.3346558, -492.6683350, 499.3336182
4: -110.9259796, 361.1823425, -108.2311325, 352.7280273, -463.6539917, 469.4134521

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097238, upper bound: 2585.4096915
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097240, upper bound: 2585.4096711
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -888.6462402, 1717.1263428, -898.1440430, 1734.0505371, -2622.6967773, 2615.2700195
1: -172.2853699, 220.8836060, -173.8924408, 223.1298065, -395.4151001, 394.7760620
2: -131.0820007, 291.9400024, -132.3890381, 294.8346252, -425.9166260, 424.3290405
3: -128.2740631, 381.5789490, -129.6087036, 385.4119263, -513.6858521, 511.1876526
4: -112.6301422, 366.6830750, -113.7345810, 370.3538513, -482.9839478, 480.4176636

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097167, upper bound: 2585.4096931
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097168, upper bound: 2585.4096718
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1425.2235107, 2774.1062012, -854.7002563, 1646.8237305, -3072.0473633, 3628.8063965
1: -278.0411682, 357.1679688, -165.3543701, 212.1933594, -490.2345276, 522.5223389
2: -212.0723572, 458.2070618, -125.9512558, 280.8149719, -492.8872986, 584.1582642
3: -210.4892120, 611.7031860, -123.5260849, 366.3346558, -576.8238525, 735.2292480
4: -182.5808258, 575.5949097, -108.2311325, 352.7280273, -535.3088379, 683.8260498

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097232, upper bound: 2585.4096209
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097231, upper bound: 2585.4095927
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1434.5151367, 2795.5646973, -898.1440430, 1734.0505371, -3168.5656738, 3693.7084961
1: -280.2090149, 359.7027588, -173.8924408, 223.1298065, -503.3387451, 533.5952148
2: -213.5428467, 461.7174377, -132.3890381, 294.8346252, -508.3774719, 594.1064453
3: -211.8654938, 616.0806885, -129.6087036, 385.4119263, -597.2772827, 745.6893311
4: -183.8469391, 579.8562622, -113.7345810, 370.3538513, -554.2008057, 693.5908203

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097162, upper bound: 2585.4096200
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097162, upper bound: 2585.4095918
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -850.6367188, 1634.7388916, -1361.1849365, 2652.8679199, -3503.5046387, 2995.9228516
1: -163.9452820, 210.6856232, -265.6477051, 341.6111450, -505.5564270, 476.3333130
2: -125.0168533, 279.0747681, -202.6729584, 437.5845032, -562.6013794, 481.7477112
3: -122.3476944, 364.1907349, -200.9354401, 584.4621582, -706.8097534, 565.1261597
4: -107.3531418, 350.7477722, -174.3999634, 549.8066406, -657.1596680, 525.1477051

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097857, upper bound: 2585.4091966
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -864.1668701, 1661.9200439, -1427.5400391, 2775.4204102, -3639.5866699, 3089.4599609
1: -166.7432404, 214.1390839, -278.0932922, 357.4755249, -524.2186279, 492.2323608
2: -127.0823593, 283.8996582, -212.1349182, 459.2204590, -586.3026733, 496.0345459
3: -124.3832855, 370.1860046, -210.5332184, 612.1663818, -736.5496216, 580.7190552
4: -109.1438446, 356.7561951, -182.5993347, 576.8648071, -686.0086670, 539.3555298

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096163, upper bound: 2585.4091939
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096163, upper bound: 2585.4091939
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -886.8914795, 1709.8226318, -1367.5848389, 2668.7744141, -3555.6655273, 3077.4074707
1: -171.2387848, 219.9608002, -267.2972717, 343.4701233, -514.7089233, 487.2580566
2: -130.4470520, 291.5432129, -203.7330780, 440.2134705, -570.6605225, 495.2762146
3: -127.4320221, 380.4510498, -201.8130188, 587.7340698, -715.1660767, 582.2640381
4: -111.9680634, 366.3427429, -175.2687073, 553.0568237, -665.0247803, 541.6114502

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -906.3043213, 1747.0827637, -1437.0803223, 2796.8210449, -3703.1254883, 3184.1630859
1: -175.0099487, 224.7739563, -280.2695312, 360.0525818, -535.0625000, 505.0434875
2: -133.3369293, 298.1164551, -213.6444397, 462.5255432, -595.8624878, 511.7608032
3: -130.3515625, 388.8265076, -211.9433136, 616.4910889, -746.8426514, 600.7698364
4: -114.4876022, 374.5560608, -183.8939667, 580.9580688, -695.4456787, 558.4500122

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096335, upper bound: 2585.4095932
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096335, upper bound: 2585.4095912
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1007.3342285, 1931.8604736, -857.8508301, 1647.0145264, -2654.3486328, 2789.7106934
1: -193.7399597, 249.2700806, -165.2586517, 212.4072723, -406.1471863, 414.5287476
2: -148.1012573, 329.7397766, -126.0152130, 281.6795044, -429.7807617, 455.7550049
3: -146.4978638, 430.6784973, -123.7115707, 366.8823242, -513.3800659, 554.3899536
4: -127.4615021, 414.2211609, -108.2900391, 353.8258972, -481.2873840, 522.5112305

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097266, upper bound: 2585.4098490
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097242, upper bound: 2585.4097161
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1025.8618164, 1966.2758789, -901.8145142, 1736.4880371, -2762.3498535, 2868.0903320
1: -197.1512299, 253.7752228, -174.1793976, 223.6518097, -420.8030396, 427.9546204
2: -150.8016052, 334.9891052, -132.6672211, 295.7603760, -446.5619812, 467.6562500
3: -149.3534851, 438.4360352, -129.9446259, 386.2217712, -535.5752563, 568.3806763
4: -129.8260498, 420.7496643, -113.9591599, 371.6158752, -501.4419250, 534.7088013

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097193, upper bound: 2585.4098499
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097171, upper bound: 2585.4097149
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1562.4378662, 3021.1933594, -857.8508301, 1647.0145264, -3209.4523926, 3879.0437012
1: -302.8604736, 389.4263000, -165.2586517, 212.4072723, -515.2677612, 554.6849365
2: -231.4158936, 501.6635742, -126.0152130, 281.6795044, -513.0953369, 627.6787720
3: -231.1779022, 668.1118164, -123.7115707, 366.8823242, -598.0602417, 791.8233643
4: -199.4640503, 629.8930054, -108.2900391, 353.8258972, -553.2898560, 738.1830444

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097261, upper bound: 2585.4098172
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097237, upper bound: 2585.4096337
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1580.2353516, 3054.6037598, -901.8145142, 1736.4880371, -3316.7231445, 3956.4182129
1: -306.1796875, 393.7770386, -174.1793976, 223.6518097, -529.8314819, 567.9564209
2: -234.0107269, 506.7441101, -132.6672211, 295.7603760, -529.7711182, 639.4113159
3: -233.9362640, 675.6757812, -129.9446259, 386.2217712, -620.1579590, 805.6203613
4: -201.7374878, 636.2158203, -113.9591599, 371.6158752, -573.3533936, 750.1749878

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098216
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098216
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -956.9135742, 1831.8260498, -1364.0747070, 2655.1252441, -3612.0388184, 3195.9003906
1: -183.6010742, 236.5175018, -265.7996521, 341.9778748, -525.5789795, 502.3171082
2: -140.3819733, 314.3294983, -202.8582001, 438.4284668, -578.8104248, 517.1876221
3: -138.3016052, 408.8683472, -201.2136993, 585.1735229, -723.4750977, 610.0820312
4: -120.6863098, 394.9391479, -174.5583954, 550.8916016, -671.5777588, 569.4974365

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096279
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -983.8361816, 1883.6821289, -1432.4635010, 2781.4245605, -3765.2602539, 3316.1455078
1: -188.8506012, 243.2518005, -278.6390991, 358.3676147, -547.2182007, 521.8908691
2: -144.4063873, 323.3523560, -212.6272125, 460.6875610, -605.0939331, 535.9795532
3: -142.4181366, 420.4886780, -211.1055603, 613.7294312, -756.1475220, 631.5941162
4: -124.2017212, 406.2228699, -183.0257111, 578.7135620, -702.9152222, 589.2485962

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096333, upper bound: 2585.4096275
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096333, upper bound: 2585.4096274
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1058.2601318, 2023.6773682, -1407.6361084, 2731.2802734, -3789.5405273, 3431.3134766
1: -202.6812744, 261.3731384, -273.5722656, 352.0185242, -554.6996460, 534.9454346
2: -155.2660217, 345.3315125, -208.9079285, 452.5324402, -607.7984619, 554.2393188
3: -153.4738464, 451.8710022, -207.6834259, 603.0379639, -756.5116577, 659.5543823
4: -133.5735931, 433.8233948, -179.9340057, 568.4255981, -701.9990845, 613.7573242

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1058.2601318, 2023.6773682, -1443.1108398, 2810.8596191, -3869.1196289, 3466.7880859
1: -202.6812744, 261.3731384, -281.7601013, 361.6717529, -564.3530273, 543.1332397
2: -155.2660217, 345.3315125, -214.6348724, 465.0590820, -620.3250732, 559.9663086
3: -153.4738464, 451.8710022, -212.8840637, 619.5979614, -773.0716553, 664.7550049
4: -133.5735931, 433.8233948, -184.7601013, 584.0769043, -717.6504517, 618.5834961

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.95 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097238, upper bound: 2585.4096915
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097240, upper bound: 2585.4096711
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097167, upper bound: 2585.4096931
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097168, upper bound: 2585.4096718
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097232, upper bound: 2585.4096209
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097231, upper bound: 2585.4095927
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097162, upper bound: 2585.4096200
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097162, upper bound: 2585.4095918
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097857, upper bound: 2585.4091966
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4096163, upper bound: 2585.4091939
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4096163, upper bound: 2585.4091939
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4096335, upper bound: 2585.4095932
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4096335, upper bound: 2585.4095912
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097266, upper bound: 2585.4098490
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097242, upper bound: 2585.4097161
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097193, upper bound: 2585.4098499
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097171, upper bound: 2585.4097149
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097261, upper bound: 2585.4098172
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4097237, upper bound: 2585.4096337
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098216
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098531, upper bound: 2585.4098216
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096279
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4096333, upper bound: 2585.4096275
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4096333, upper bound: 2585.4096274
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.95
Output dim: 0, lower bound: -2585.4098214, upper bound: 2585.4098214

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -782.9882202, 1517.0867920, -827.2229004, 1593.6794434, -2376.6672363, 2344.3095703
1: -152.3001099, 195.0890045, -160.0129852, 205.3742065, -357.6742859, 355.1019897
2: -115.7664566, 257.2363586, -121.8876572, 271.5256653, -387.2921143, 379.1240234
3: -113.1237411, 336.5605774, -119.4859619, 354.4726868, -467.5964355, 456.0465393
4: -99.3923645, 323.1150818, -104.7235794, 341.1410522, -440.5334167, 427.8386536

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095417, upper bound: 2585.4090919
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094073, upper bound: 2585.4090920
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -843.1072388, 1630.2143555, -843.8176880, 1626.6289062, -2469.7360840, 2474.0317383
1: -163.6396332, 209.6610413, -163.3406372, 209.5515442, -373.1911621, 373.0015564
2: -124.4100723, 277.0109253, -124.3774643, 277.3145142, -401.7245789, 401.3883362
3: -121.5922928, 361.9627380, -121.9500809, 361.7505798, -483.3428650, 483.9128113
4: -106.8357773, 347.9775391, -106.8725967, 348.3494873, -455.1852417, 454.8501282

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4070292, upper bound: 2585.4077906
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097115, upper bound: 2585.4096726
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -793.8908691, 1538.4398193, -867.3862915, 1675.0189209, -2468.9096680, 2405.8256836
1: -154.3845825, 197.7843475, -167.9800720, 215.5180664, -369.9026184, 365.7644043
2: -117.3481140, 260.8364868, -127.8395462, 284.5251770, -401.8732605, 388.6760254
3: -114.7396393, 341.3545837, -125.0572128, 372.1831360, -486.9227905, 466.4117737
4: -100.7941132, 327.7143555, -109.8023529, 357.4740906, -458.2681580, 437.5166931

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095739, upper bound: 2585.4090938
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093040, upper bound: 2585.4090890
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -856.7895508, 1657.2635498, -888.9006348, 1716.5634766, -2573.3530273, 2546.1640625
1: -166.2823792, 213.1284637, -172.1424866, 220.8605347, -387.1428833, 385.2709351
2: -126.4532394, 281.5469055, -131.0408630, 291.7989502, -418.2521667, 412.5877686
3: -123.7145538, 368.0685730, -128.2747650, 381.4635620, -505.1781006, 496.3433228
4: -108.6405258, 353.7158508, -112.5714035, 366.5641785, -475.2046204, 466.2872620

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095916, upper bound: 2585.4090938
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093111, upper bound: 2585.4090890
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1339.4816895, 2612.8354492, -827.2229004, 1593.6794434, -2933.1608887, 3440.0583496
1: -261.7186890, 336.3645020, -160.0129852, 205.3742065, -467.0928955, 496.3775024
2: -199.5876312, 430.4178467, -121.8876572, 271.5256653, -471.1132812, 552.3054810
3: -197.8846893, 575.4683228, -119.4859619, 354.4726868, -552.3573608, 694.9542236
4: -171.7608337, 540.7949829, -104.7235794, 341.1410522, -512.9018555, 645.5184937

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095486, upper bound: 2585.4091002
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094271, upper bound: 2585.4091002
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1399.7562256, 2725.1975098, -843.8176880, 1626.6289062, -3026.3852539, 3569.0148926
1: -273.1458130, 350.8142090, -163.3406372, 209.5515442, -482.6973572, 514.1548462
2: -208.2378845, 449.8674622, -124.3774643, 277.3145142, -485.5523987, 574.2449341
3: -206.6479187, 600.7561035, -121.9500809, 361.7505798, -568.3983765, 722.7061768
4: -179.2646942, 565.1715698, -106.8725967, 348.3494873, -527.6141357, 672.0440674

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095601, upper bound: 2585.4091002
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094320, upper bound: 2585.4091002
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1346.4737549, 2630.2670898, -867.3862915, 1675.0189209, -3021.4926758, 3497.6530762
1: -263.4957886, 338.3715820, -167.9800720, 215.5180664, -479.0138245, 506.3516541
2: -200.7264252, 433.4999695, -127.8395462, 284.5251770, -485.2515869, 561.3395386
3: -198.8496857, 579.0509644, -125.0572128, 372.1831360, -571.0328369, 704.1080933
4: -172.7048492, 544.6254272, -109.8023529, 357.4740906, -530.1789551, 654.4277954

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095822, upper bound: 2585.4091021
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093040, upper bound: 2585.4090989
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1410.1641846, 2748.1345215, -888.9006348, 1716.5634766, -3126.7275391, 3637.0351562
1: -275.4554749, 353.5707092, -172.1424866, 220.8605347, -496.3160095, 525.7131958
2: -209.8444672, 453.6001587, -131.0408630, 291.7989502, -501.6434021, 584.6409912
3: -208.2342377, 605.5097656, -128.2747650, 381.4635620, -589.6978149, 733.7845459
4: -180.6724396, 569.7161255, -112.5714035, 366.5641785, -547.2365112, 682.2874756

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095911, upper bound: 2585.4091021
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093106, upper bound: 2585.4090989
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -801.3940430, 1548.7037354, -1361.1849365, 2652.8679199, -3454.2617188, 2909.8879395
1: -155.5769653, 199.3390045, -265.6477051, 341.6111450, -497.1881104, 464.9866943
2: -118.3499222, 263.1139221, -202.6729584, 437.5845032, -555.9343262, 465.7868652
3: -115.9237289, 343.9761658, -200.9354401, 584.4621582, -700.3858032, 544.9115601
4: -101.7017975, 330.5035706, -174.3999634, 549.8066406, -651.5084229, 504.9035034

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1355.5399170, 2636.7272949, -1361.1849365, 2652.8679199, -4007.1452637, 3996.8681641
1: -264.1822815, 339.5686035, -265.6477051, 341.6111450, -604.1582031, 603.3896484
2: -201.6161957, 434.8338318, -202.6729584, 437.5845032, -639.2006836, 637.5067139
3: -200.3197784, 581.5281982, -200.9354401, 584.4621582, -784.7819214, 782.4636230
4: -173.6524506, 546.3600464, -174.3999634, 549.8066406, -723.4591064, 720.7600098

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -819.0875854, 1583.8397217, -1427.5400391, 2775.4204102, -3594.5078125, 3011.3798828
1: -159.1192322, 203.8007355, -278.0932922, 357.4755249, -516.5946655, 481.8940430
2: -121.0074692, 269.3359375, -212.1349182, 459.2204590, -580.2279053, 481.4708557
3: -118.5498886, 351.7332764, -210.5332184, 612.1663818, -730.7162476, 562.2661743
4: -103.9928513, 338.2415161, -182.5993347, 576.8648071, -680.8576660, 520.8408203

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096100, upper bound: 2585.4091930
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096100, upper bound: 2585.4091939
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1377.5933838, 2678.2802734, -1427.5400391, 2775.4204102, -4152.6132812, 4105.1289062
1: -268.4143677, 344.9464111, -278.0932922, 357.4755249, -624.3986816, 621.2797852
2: -204.8221588, 442.1853943, -212.1349182, 459.2204590, -664.0426025, 654.3201294
3: -203.5622253, 590.8854370, -210.5332184, 612.1663818, -815.7286377, 801.4184570
4: -176.4319916, 555.5137329, -182.5993347, 576.8648071, -753.2968140, 738.1130371

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096099, upper bound: 2585.4091930
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096100, upper bound: 2585.4091950
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -838.8217773, 1623.6011963, -1367.5848389, 2668.7744141, -3507.5959473, 2991.1860352
1: -162.8777771, 208.7070312, -267.2972717, 343.4701233, -506.3478394, 476.0043030
2: -123.8529282, 275.5956116, -203.7330780, 440.2134705, -564.0662842, 479.3286743
3: -121.1648712, 360.5318298, -201.8130188, 587.7340698, -708.8988647, 562.3448486
4: -106.4020615, 346.2711792, -175.2687073, 553.0568237, -659.4587402, 521.5399170

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095935
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1379.6035156, 2696.6477051, -1367.5848389, 2668.7744141, -4048.3779297, 4064.2324219
1: -270.3161011, 346.5545654, -267.2972717, 343.4701233, -612.5048218, 612.9252930
2: -205.7032623, 444.6619873, -203.7330780, 440.2134705, -645.9166260, 648.3949585
3: -203.8887329, 593.6409302, -201.8130188, 587.7340698, -791.6226807, 795.4539185
4: -177.0714569, 558.4535522, -175.2687073, 553.0568237, -730.1282349, 733.7221680

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095935
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -861.4136963, 1666.8903809, -1437.0803223, 2796.8210449, -3658.2343750, 3103.9707031
1: -167.1999817, 214.2951508, -280.2695312, 360.0525818, -527.2525635, 494.5646667
2: -127.1888046, 283.2002869, -213.6444397, 462.5255432, -589.7142944, 496.8447266
3: -124.5308838, 370.2161865, -211.9433136, 616.4910889, -741.0219727, 582.1594238
4: -109.2905197, 355.7697449, -183.8939667, 580.9580688, -690.2485352, 539.6635742

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095902
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095912
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1407.2047119, 2747.6892090, -1437.0803223, 2796.8210449, -4204.0258789, 4184.7695312
1: -275.4772034, 353.1534119, -280.2695312, 360.0525818, -634.3923340, 632.5567017
2: -209.6806335, 453.6230164, -213.6444397, 462.5255432, -672.2061768, 667.2674561
3: -208.0089569, 605.2241821, -211.9433136, 616.4910889, -824.5000610, 817.1674805
4: -180.5376129, 569.6154175, -183.8939667, 580.9580688, -761.4956665, 753.5092773

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095918
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095931
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -891.0786133, 1714.1137695, -831.1973267, 1595.8338623, -2486.9125977, 2545.3110352
1: -171.9225922, 221.0413513, -160.0706787, 205.8104706, -377.7330322, 381.1120300
2: -131.3336182, 291.0024109, -122.0818710, 272.6549072, -403.9885254, 413.0842896
3: -129.6641388, 381.4725647, -119.8001862, 355.3925171, -485.0566406, 501.2727661
4: -112.9494247, 365.7009888, -104.8973846, 342.5775452, -455.5268860, 470.5983582

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095600, upper bound: 2585.4092661
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094306, upper bound: 2585.4092674
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -983.6187134, 1887.0545654, -846.4826660, 1625.5317383, -2609.1499023, 2733.5371094
1: -189.2561798, 243.4794159, -163.1292877, 209.6166229, -398.8727722, 406.6087036
2: -144.6494141, 322.0057373, -124.3524323, 278.0032349, -422.6526489, 446.3581543
3: -143.1150055, 420.5931702, -122.0560074, 362.0540771, -505.1690063, 542.6491699
4: -124.4905014, 404.5077515, -106.8559341, 349.2289124, -473.7194214, 511.3636780

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071030, upper bound: 2585.4078753
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097142, upper bound: 2585.4097155
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -909.5593872, 1747.3256836, -871.6917114, 1678.7235107, -2588.2829590, 2619.0168457
1: -175.2010193, 225.4306641, -168.3793030, 216.1937103, -391.3947144, 393.8099060
2: -133.9560852, 296.0624695, -128.2056274, 285.7676392, -419.7236938, 424.2680969
3: -132.3971863, 389.0001221, -125.4615784, 373.2653198, -505.6625061, 514.4616089
4: -115.2358398, 371.9818115, -110.0960388, 359.1359253, -474.3717651, 482.0778503

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095886, upper bound: 2585.4092679
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093094, upper bound: 2585.4092649
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1003.6065063, 1924.1279297, -891.9882202, 1718.0100098, -2721.6164551, 2816.1162109
1: -192.9225922, 248.3268890, -172.3384705, 221.2538147, -414.1763611, 420.6652527
2: -147.5606079, 327.7014465, -131.2437592, 292.5313416, -440.0919495, 458.9451904
3: -146.1614380, 428.9334106, -128.5251923, 382.0405884, -528.2020264, 557.4586182
4: -127.0324097, 411.5915527, -112.7307587, 367.5705261, -494.6029358, 524.3223267

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095942, upper bound: 2585.4093168
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093142, upper bound: 2585.4093123
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1457.5163574, 2825.3388672, -831.1973267, 1595.8338623, -3053.3500977, 3656.5358887
1: -283.0638428, 363.9477539, -160.0706787, 205.8104706, -488.8742981, 524.0183716
2: -216.1905212, 467.3753662, -122.0818710, 272.6549072, -488.8454285, 589.4572144
3: -215.6173706, 624.0400391, -119.8001862, 355.3925171, -571.0097656, 743.8402100
4: -186.2228088, 587.0644531, -104.8973846, 342.5775452, -528.8003540, 691.9618530

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095585, upper bound: 2585.4092672
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094270, upper bound: 2585.4092688
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1542.9008789, 2982.0544434, -846.4826660, 1625.5317383, -3168.4313965, 3828.5371094
1: -298.9465942, 384.4208984, -163.1292877, 209.6166229, -508.5631714, 547.5501709
2: -228.4027710, 494.8878784, -124.3524323, 278.0032349, -506.4060059, 619.2402954
3: -228.2973633, 659.5048828, -122.0560074, 362.0540771, -590.3514404, 781.5609131
4: -196.8845825, 621.3804321, -106.8559341, 349.2289124, -546.1135254, 728.2363892

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095633, upper bound: 2585.4093124
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094351, upper bound: 2585.4093163
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1507.4355469, 2917.0859375, -901.8145142, 1736.4880371, -3243.9235840, 3818.9003906
1: -292.5619812, 375.9746704, -174.1793976, 223.6518097, -516.2138062, 550.1540527
2: -223.4577484, 484.6454163, -132.6672211, 295.7603760, -519.2181396, 617.3126221
3: -223.1496277, 645.0739746, -129.9446259, 386.2217712, -609.3713379, 775.0185547
4: -192.6057129, 608.4564819, -113.9591599, 371.6158752, -564.2214966, 722.4155884

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097188, upper bound: 2585.4093189
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093141, upper bound: 2585.4093129
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1568.8863525, 3032.4340820, -901.8145142, 1736.4880371, -3305.3735352, 3934.2485352
1: -304.0259399, 390.9632568, -174.1793976, 223.6518097, -527.6777344, 565.1426392
2: -232.3823395, 502.1742249, -132.6672211, 295.7603760, -528.1427002, 634.8414307
3: -232.3887177, 670.7070312, -129.9446259, 386.2217712, -618.6103516, 800.6516113
4: -200.3506165, 630.3029785, -113.9591599, 371.6158752, -571.9664307, 744.2621460

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097188, upper bound: 2585.4093208
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093140, upper bound: 2585.4093144
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -909.8465576, 1749.6484375, -1364.0747070, 2655.1252441, -3564.9716797, 3113.7231445
1: -175.6346283, 225.6358795, -265.7996521, 341.9778748, -517.6124878, 491.4355164
2: -134.0610962, 298.9619751, -202.8582001, 438.4284668, -572.4895630, 501.8201904
3: -132.3726196, 389.5834351, -201.2136993, 585.1735229, -717.5461426, 590.7971191
4: -115.3436356, 375.4966125, -174.5583954, 550.8916016, -666.2352295, 550.0549927

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1469.6578369, 2845.9616699, -1364.0747070, 2655.1252441, -4124.6582031, 4209.5068359
1: -285.3706665, 366.7057800, -265.7996521, 341.9778748, -625.6561890, 630.9354248
2: -217.9227753, 472.2869873, -202.8582001, 438.4284668, -656.3512573, 675.1452026
3: -217.4987030, 629.0374146, -201.2136993, 585.1735229, -802.6721802, 830.2510986
4: -187.8004761, 593.0531006, -174.5583954, 550.8916016, -738.6920166, 767.6114502

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -939.6300049, 1806.9371338, -1432.4635010, 2781.4245605, -3721.0539551, 3239.4006348
1: -181.3768463, 233.0354614, -278.6390991, 358.3676147, -539.7444458, 511.6744690
2: -138.4805450, 308.9454041, -212.6272125, 460.6875610, -599.1680908, 521.5726318
3: -136.8685303, 402.4082947, -211.1055603, 613.7294312, -750.5979614, 613.5137939
4: -119.1783142, 387.9396362, -183.0257111, 578.7135620, -697.8917847, 570.9653320

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096269
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096275
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1502.1312256, 2906.5795898, -1432.4635010, 2781.4245605, -4283.5556641, 4339.0419922
1: -291.5064697, 374.6307373, -278.6390991, 358.3676147, -648.3662109, 651.7761230
2: -222.6384888, 482.8501282, -212.6272125, 460.6875610, -683.3260498, 695.4772949
3: -222.3642578, 642.7557373, -211.1055603, 613.7294312, -836.0936890, 853.8612671
4: -191.9031067, 606.2064819, -183.0257111, 578.7135620, -770.6166992, 789.2321167

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096269
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096269
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1013.6481934, 1943.2452393, -1407.6361084, 2731.2802734, -3744.9284668, 3350.8811035
1: -194.8709869, 250.8099365, -273.5722656, 352.0185242, -546.8893433, 524.3822021
2: -149.0917206, 330.1845093, -208.9079285, 452.5324402, -601.6241455, 539.0922241
3: -147.7005463, 433.1069946, -207.6834259, 603.0379639, -750.7385254, 640.7904053
4: -128.3608704, 414.5717468, -179.9340057, 568.4255981, -696.7864380, 594.5057373

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100634, upper bound: 2585.4096362
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096274, upper bound: 2585.4096333
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1568.8863525, 3032.4340820, -1407.6361084, 2731.2802734, -4299.6821289, 4438.3017578
1: -304.0259399, 390.9632568, -273.5722656, 352.0185242, -654.2468262, 663.0364380
2: -232.3823395, 502.1742249, -208.9079285, 452.5324402, -684.9147949, 711.0820923
3: -232.3887177, 670.7070312, -207.6834259, 603.0379639, -835.4265747, 878.3904419
4: -200.3506165, 630.3029785, -179.9340057, 568.4255981, -768.7761230, 810.2369385

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4100633, upper bound: 2585.4096362
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096274, upper bound: 2585.4096313
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1013.6481934, 1943.2452393, -1443.1108398, 2810.8596191, -3824.5078125, 3386.3559570
1: -194.8709869, 250.8099365, -281.7601013, 361.6717529, -556.5427246, 532.5700684
2: -149.0917206, 330.1845093, -214.6348724, 465.0590820, -614.1508179, 544.8192139
3: -147.7005463, 433.1069946, -212.8840637, 619.5979614, -767.2985229, 645.9910278
4: -128.3608704, 414.5717468, -184.7601013, 584.0769043, -712.4377441, 599.3318481

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098174, upper bound: 2585.4096366
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096340, upper bound: 2585.4096340
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1568.8863525, 3032.4340820, -1443.1108398, 2810.8596191, -4379.7460938, 4475.5444336
1: -304.0259399, 390.9632568, -281.7601013, 361.6717529, -664.8048096, 671.4874268
2: -232.3823395, 502.1742249, -214.6348724, 465.0590820, -697.4414062, 716.8090820
3: -232.3887177, 670.7070312, -212.8840637, 619.5979614, -851.9866333, 883.5910645
4: -200.3506165, 630.3029785, -184.7601013, 584.0769043, -784.4274902, 815.0631104

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098174, upper bound: 2585.4096366
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096340
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.64 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095417, upper bound: 2585.4090919
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4094073, upper bound: 2585.4090920
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4070292, upper bound: 2585.4077906
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097115, upper bound: 2585.4096726
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095739, upper bound: 2585.4090938
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093040, upper bound: 2585.4090890
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095916, upper bound: 2585.4090938
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093111, upper bound: 2585.4090890
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095486, upper bound: 2585.4091002
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4094271, upper bound: 2585.4091002
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095601, upper bound: 2585.4091002
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4094320, upper bound: 2585.4091002
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095822, upper bound: 2585.4091021
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093040, upper bound: 2585.4090989
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095911, upper bound: 2585.4091021
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093106, upper bound: 2585.4090989
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097861, upper bound: 2585.4091981
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096100, upper bound: 2585.4091930
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096100, upper bound: 2585.4091939
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096099, upper bound: 2585.4091930
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096100, upper bound: 2585.4091950
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095935
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095935
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095902
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095912
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095918
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095931
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095600, upper bound: 2585.4092661
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4094306, upper bound: 2585.4092674
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4071030, upper bound: 2585.4078753
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097142, upper bound: 2585.4097155
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095886, upper bound: 2585.4092679
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093094, upper bound: 2585.4092649
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095942, upper bound: 2585.4093168
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093142, upper bound: 2585.4093123
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095585, upper bound: 2585.4092672
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4094270, upper bound: 2585.4092688
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4095633, upper bound: 2585.4093124
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4094351, upper bound: 2585.4093163
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097188, upper bound: 2585.4093189
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093141, upper bound: 2585.4093129
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4097188, upper bound: 2585.4093208
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4093140, upper bound: 2585.4093144
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4096302
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096269
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096275
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096269
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096269
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4100634, upper bound: 2585.4096362
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096274, upper bound: 2585.4096333
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4100633, upper bound: 2585.4096362
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096274, upper bound: 2585.4096313
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098174, upper bound: 2585.4096366
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096340, upper bound: 2585.4096340
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4098174, upper bound: 2585.4096366
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4096340

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -782.9882202, 1517.0867920, -807.1816406, 1555.7858887, -2338.7739258, 2324.2683105
1: -152.3001099, 195.0890045, -156.2143097, 200.4934998, -352.7936096, 351.3033142
2: -115.7664566, 257.2363586, -118.9697037, 264.7739258, -380.5403748, 376.2060547
3: -113.1237411, 336.5605774, -116.5758057, 345.8892517, -459.0130005, 453.1363220
4: -99.3923645, 323.1150818, -102.2187042, 332.7384033, -432.1307373, 425.3337708

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094739, upper bound: 2585.4090204
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094739, upper bound: 2585.4090896
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -782.9882202, 1517.0867920, -843.7017822, 1622.8280029, -2405.8161621, 2360.7885742
1: -152.3001099, 195.0890045, -162.7903900, 209.2380676, -361.5380859, 357.8793945
2: -115.7664566, 257.2363586, -124.2565689, 275.4249268, -391.1913452, 381.4929199
3: -113.1237411, 336.5605774, -121.9161224, 360.8500671, -473.9738159, 458.4766541
4: -99.3923645, 323.1150818, -106.7682190, 346.0500183, -445.4423828, 429.8832397

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093302, upper bound: 2585.4090224
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093303, upper bound: 2585.4090919
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -818.6004639, 1582.7268066, -774.9073486, 1499.1608887, -2317.7612305, 2357.6337891
1: -158.9296875, 203.4833374, -150.9895935, 192.9426880, -351.8723755, 354.4729309
2: -120.7527618, 268.4933472, -114.4517136, 254.0246277, -374.7773438, 382.9450073
3: -118.0717545, 351.1979370, -111.9814987, 332.2784424, -450.3501892, 463.1793823
4: -103.6926346, 337.3613892, -98.2585983, 319.1926880, -422.8853149, 435.6199951

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4070292, upper bound: 2585.4077883
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4070292, upper bound: 2585.4077883
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -843.1072388, 1630.2143555, -834.4132080, 1610.0970459, -2453.2043457, 2464.6274414
1: -163.6396332, 209.6610413, -161.7000732, 207.2904358, -370.9300537, 371.3610840
2: -124.4100723, 277.0109253, -123.0542984, 274.3687134, -398.7787781, 400.0652161
3: -121.5922928, 361.9627380, -120.6962738, 357.8971558, -479.4894409, 482.6589966
4: -106.8357773, 347.9775391, -105.7611237, 344.6132507, -451.4489746, 453.7386475

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096973, upper bound: 2585.4093016
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096977, upper bound: 2585.4096726
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -793.8908691, 1538.4398193, -847.2018433, 1637.4080811, -2431.2988281, 2385.6416016
1: -154.3845825, 197.7843475, -164.2238922, 210.6356964, -365.0202637, 362.0081787
2: -117.3481140, 260.8364868, -124.9070816, 277.8570557, -395.2050781, 385.7435608
3: -114.7396393, 341.3545837, -122.0996933, 363.5687256, -478.3083496, 463.4542236
4: -100.7941132, 327.7143555, -107.2727127, 349.1568298, -449.9508667, 434.9870605

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094739, upper bound: 2585.4090238
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094739, upper bound: 2585.4090902
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -793.8908691, 1538.4398193, -890.3657227, 1716.4387207, -2510.3295898, 2428.8054199
1: -154.3845825, 197.7843475, -171.9631042, 220.9271698, -375.3117065, 369.7474365
2: -117.3481140, 260.8364868, -131.1069336, 290.5997009, -407.9478149, 391.9434204
3: -114.7396393, 341.3545837, -128.2950287, 381.3274841, -496.0671387, 469.6495972
4: -100.7941132, 327.7143555, -112.5892029, 365.1167297, -465.9107666, 440.3034973

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4092825, upper bound: 2585.4090222
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4092825, upper bound: 2585.4090908
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -856.7895508, 1657.2635498, -868.3173218, 1678.2119141, -2535.0014648, 2525.5805664
1: -166.2823792, 213.1284637, -168.3173828, 215.8850555, -382.1673889, 381.4458313
2: -126.4532394, 281.5469055, -128.0528412, 285.0067444, -411.4599915, 409.5997314
3: -123.7145538, 368.0685730, -125.2705536, 372.6880798, -496.4026184, 493.3391113
4: -108.6405258, 353.7158508, -109.9979477, 358.0870972, -466.7275391, 463.7138062

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095553, upper bound: 2585.4089426
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095556, upper bound: 2585.4090925
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -856.7895508, 1657.2635498, -908.9724121, 1752.8483887, -2609.6376953, 2566.2358398
1: -166.2823792, 213.1284637, -175.6114960, 225.6242218, -391.9066162, 388.7398987
2: -126.4532394, 281.5469055, -133.9141083, 297.0107117, -423.4639587, 415.4609375
3: -123.7145538, 368.0685730, -131.1008606, 389.4499207, -513.1644897, 499.1694336
4: -108.6405258, 353.7158508, -115.0181046, 373.1087036, -481.7491455, 468.7339478

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093045, upper bound: 2585.4089402
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093045, upper bound: 2585.4090908
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1339.4261475, 2612.7170410, -807.1816406, 1555.7858887, -2895.2116699, 3419.8984375
1: -261.7065125, 336.3496399, -156.2143097, 200.4934998, -462.2000122, 492.5639343
2: -199.5788269, 430.3989868, -118.9697037, 264.7739258, -464.3527527, 549.3686523
3: -197.8760376, 575.4432373, -116.5758057, 345.8892517, -543.7652588, 692.0190430
4: -171.7531738, 540.7711182, -102.2187042, 332.7384033, -504.4915466, 642.9896851

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094660, upper bound: 2585.4090223
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094660, upper bound: 2585.4091002
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1339.5128174, 2612.9020996, -843.7017822, 1622.8280029, -2962.3408203, 3456.6037598
1: -261.7254944, 336.3728638, -162.7903900, 209.2380676, -470.9634705, 499.1632690
2: -199.5926056, 430.4284363, -124.2565689, 275.4249268, -475.0175171, 554.6849976
3: -197.8895264, 575.4825439, -121.9161224, 360.8500671, -558.7395630, 697.3985596
4: -171.7651367, 540.8082886, -106.7682190, 346.0500183, -517.8151855, 647.5763550

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093257, upper bound: 2585.4090224
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093257, upper bound: 2585.4091002
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1399.7562256, 2725.1975098, -823.8435669, 1588.6127930, -2988.3691406, 3549.0407715
1: -273.1458130, 350.8142090, -159.5160828, 204.6628418, -477.8086243, 510.3302612
2: -208.2378845, 449.8674622, -121.4479370, 270.5566711, -478.7945557, 571.3154297
3: -206.6479187, 600.7561035, -119.0337677, 353.1546936, -559.8025513, 719.7898560
4: -179.2646942, 565.1715698, -104.3525162, 339.9537964, -519.2184448, 669.5241089

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095517, upper bound: 2585.4089410
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095521, upper bound: 2585.4090979
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1399.7562256, 2725.1975098, -857.2885132, 1649.9844971, -3049.7407227, 3582.4860840
1: -273.1458130, 350.8142090, -165.5368042, 212.6811829, -485.8269958, 516.3509521
2: -208.2378845, 449.8674622, -126.3077011, 280.2595825, -488.4974670, 576.1751099
3: -206.6479187, 600.7561035, -123.9425735, 366.8183594, -573.4662476, 724.6986694
4: -179.2646942, 565.1715698, -108.5259399, 352.0743713, -531.3388672, 673.6974487

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093257, upper bound: 2585.4089408
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094236, upper bound: 2585.4091014
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1346.4737549, 2630.2670898, -847.2018433, 1637.4080811, -2983.8815918, 3477.4689941
1: -263.4957886, 338.3715820, -164.2238922, 210.6356964, -474.1314697, 502.5953979
2: -200.7264252, 433.4999695, -124.9070816, 277.8570557, -478.5834656, 558.4069824
3: -198.8496857, 579.0509644, -122.0996933, 363.5687256, -562.4183960, 701.1506348
4: -172.7048492, 544.6254272, -107.2727127, 349.1568298, -521.8616333, 651.8981323

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094660, upper bound: 2585.4090239
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094660, upper bound: 2585.4091009
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1346.4737549, 2630.2670898, -890.3657227, 1716.4387207, -3062.9125977, 3520.6328125
1: -263.4957886, 338.3715820, -171.9631042, 220.9271698, -484.4229126, 510.3346558
2: -200.7264252, 433.4999695, -131.1069336, 290.5997009, -491.3261108, 564.6068726
3: -198.8496857, 579.0509644, -128.2950287, 381.3274841, -580.1771851, 707.3460083
4: -172.7048492, 544.6254272, -112.5892029, 365.1167297, -537.8215332, 657.2145996

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4092809, upper bound: 2585.4090222
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4092809, upper bound: 2585.4090989
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1410.1641846, 2748.1345215, -868.3173218, 1678.2119141, -3088.3759766, 3616.4516602
1: -275.4554749, 353.5707092, -168.3173828, 215.8850555, -491.3405151, 521.8880615
2: -209.8444672, 453.6001587, -128.0528412, 285.0067444, -494.8511963, 581.6530151
3: -208.2342377, 605.5097656, -125.2705536, 372.6880798, -580.9223022, 730.7803345
4: -180.6724396, 569.7161255, -109.9979477, 358.0870972, -538.7594604, 679.7140503

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095520, upper bound: 2585.4089427
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094660, upper bound: 2585.4091010
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1410.1641846, 2748.1345215, -908.9724121, 1752.8483887, -3163.0126953, 3657.1069336
1: -275.4554749, 353.5707092, -175.6114960, 225.6242218, -501.0797119, 529.1821899
2: -209.8444672, 453.6001587, -133.9141083, 297.0107117, -506.8551636, 587.5142822
3: -208.2342377, 605.5097656, -131.1008606, 389.4499207, -597.6841431, 736.6105957
4: -180.6724396, 569.7161255, -115.0181046, 373.1087036, -553.7810669, 684.7342529

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093025, upper bound: 2585.4089389
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093024, upper bound: 2585.4090989
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -801.3940430, 1548.7037354, -1328.4199219, 2586.2868652, -3387.6806641, 2877.1230469
1: -155.5769653, 199.3390045, -258.9644165, 333.1217651, -488.6987305, 458.3034058
2: -118.3499222, 263.1139221, -197.6669159, 426.5185852, -544.8683472, 460.7807617
3: -115.9237289, 343.9761658, -196.2422333, 570.1297607, -686.0534668, 540.2182617
4: -101.7017975, 330.5035706, -170.1799774, 535.9307251, -637.6325073, 500.6834717

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097751, upper bound: 2585.4093196
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097755, upper bound: 2585.4093211
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -801.3940430, 1548.7037354, -1345.1214600, 2630.2604980, -3431.6540527, 2893.8249512
1: -155.5769653, 199.3390045, -263.5054626, 338.1897888, -493.7667542, 462.8444519
2: -118.3499222, 263.1139221, -200.5605621, 433.7422180, -552.0921021, 463.6744995
3: -115.9237289, 343.9761658, -198.4790802, 578.8645630, -694.7882080, 542.4552002
4: -101.7017975, 330.5035706, -172.5252075, 544.8782349, -646.5800171, 503.0287476

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097755, upper bound: 2585.4093195
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097755, upper bound: 2585.4093211
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1355.5399170, 2636.7272949, -1328.4199219, 2586.2868652, -3938.8195801, 3962.2631836
1: -264.1822815, 339.5686035, -258.9644165, 333.1217651, -595.3823853, 596.6966553
2: -201.6161957, 434.8338318, -197.6669159, 426.5185852, -628.1347656, 632.5007324
3: -200.3197784, 581.5281982, -196.2422333, 570.1297607, -770.4495239, 777.7703247
4: -173.6524506, 546.3600464, -170.1799774, 535.9307251, -709.5831909, 716.5400391

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097718, upper bound: 2585.4091981
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097718, upper bound: 2585.4091966
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1355.5399170, 2636.7272949, -1345.1214600, 2630.2604980, -3983.4233398, 3981.8483887
1: -264.1822815, 339.5686035, -263.5054626, 338.1897888, -601.0579834, 601.1718140
2: -201.6161957, 434.8338318, -200.5605621, 433.7422180, -635.3583984, 635.3943481
3: -200.3197784, 581.5281982, -198.4790802, 578.8645630, -779.1843262, 780.0072632
4: -173.6524506, 546.3600464, -172.5252075, 544.8782349, -718.5307007, 718.8852539

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097718, upper bound: 2585.4091981
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4097718, upper bound: 2585.4091981
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -819.0875854, 1583.8397217, -1381.2449951, 2683.0236816, -3502.1113281, 2965.0847168
1: -159.1192322, 203.8007355, -268.8007507, 345.6904907, -504.8097229, 472.6015015
2: -121.0074692, 269.3359375, -205.1590729, 443.7971191, -564.8045654, 474.4949951
3: -118.5498886, 351.7332764, -203.8620911, 592.0875244, -710.6373901, 555.5951538
4: -103.9928513, 338.2415161, -176.6902771, 557.5167236, -661.5095825, 514.9317627

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095434, upper bound: 2585.4093154
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095434, upper bound: 2585.4093147
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -819.0875854, 1583.8397217, -1417.1152344, 2763.6882324, -3582.7758789, 3000.9545898
1: -159.1192322, 203.8007355, -277.0158691, 355.3963013, -514.5155029, 480.8165894
2: -121.0074692, 269.3359375, -210.9283600, 456.5484924, -577.5559692, 480.2642822
3: -118.5498886, 351.7332764, -209.2034912, 608.9016113, -727.4514771, 560.9365234
4: -103.9928513, 338.2415161, -181.5829773, 573.3156128, -677.3084717, 519.8244629

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095434, upper bound: 2585.4093156
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095434, upper bound: 2585.4093165
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1377.5933838, 2678.2802734, -1381.2449951, 2683.0236816, -4058.1892090, 4056.9326172
1: -268.4143677, 344.9464111, -268.8007507, 345.6904907, -612.3077393, 611.9793701
2: -204.8221588, 442.1853943, -205.1590729, 443.7971191, -648.6192627, 647.3444824
3: -203.5622253, 590.8854370, -203.8620911, 592.0875244, -795.6497803, 794.7474365
4: -176.4319916, 555.5137329, -176.6902771, 557.5167236, -733.9486084, 732.2039795

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095370, upper bound: 2585.4091930
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095370, upper bound: 2585.4091930
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1377.5933838, 2678.2802734, -1417.1152344, 2763.6882324, -4139.6855469, 4095.3950195
1: -268.4143677, 344.9464111, -277.0158691, 355.3963013, -622.6308594, 620.1280518
2: -204.8221588, 442.1853943, -210.9283600, 456.5484924, -661.3706665, 653.1136475
3: -203.5622253, 590.8854370, -209.2034912, 608.9016113, -812.4638672, 800.0888062
4: -176.4319916, 555.5137329, -181.5829773, 573.3156128, -749.7475586, 737.0966797

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095370, upper bound: 2585.4091952
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095370, upper bound: 2585.4091923
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -838.8217773, 1623.6011963, -1328.4199219, 2586.2868652, -3425.1086426, 2952.0209961
1: -162.8777771, 208.7070312, -258.9644165, 333.1217651, -495.9995422, 467.6714172
2: -123.8529282, 275.5956116, -197.6669159, 426.5185852, -550.3713989, 473.2624817
3: -121.1648712, 360.5318298, -196.2422333, 570.1297607, -691.2944946, 556.7740479
4: -106.4020615, 346.2711792, -170.1799774, 535.9307251, -642.3327637, 516.4511719

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098178, upper bound: 2585.4096752
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098178, upper bound: 2585.4096752
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -838.8217773, 1623.6011963, -1345.1214600, 2630.2604980, -3469.0820312, 2968.7226562
1: -162.8777771, 208.7070312, -263.5054626, 338.1897888, -501.0675354, 472.2124329
2: -123.8529282, 275.5956116, -200.5605621, 433.7422180, -557.5951538, 476.1561890
3: -121.1648712, 360.5318298, -198.4790802, 578.8645630, -700.0292358, 559.0109253
4: -106.4020615, 346.2711792, -172.5252075, 544.8782349, -651.2802734, 518.7963867

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098176, upper bound: 2585.4096752
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098176, upper bound: 2585.4096752
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1379.6035156, 2696.6477051, -1328.4199219, 2586.2868652, -3965.8903809, 4022.9267578
1: -270.3161011, 346.5545654, -258.9644165, 333.1217651, -601.4547729, 604.3355103
2: -205.7032623, 444.6619873, -197.6669159, 426.5185852, -632.2218018, 642.3288574
3: -203.8887329, 593.6409302, -196.2422333, 570.1297607, -774.0184326, 789.8831177
4: -177.0714569, 558.4535522, -170.1799774, 535.9307251, -713.0021973, 728.6335449

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095951
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1379.6035156, 2696.6477051, -1345.1214600, 2630.2604980, -4009.8640137, 4041.7690430
1: -270.3161011, 346.5545654, -263.5054626, 338.1897888, -607.4601440, 609.1424561
2: -205.7032623, 444.6619873, -200.5605621, 433.7422180, -639.4454956, 645.2225342
3: -203.8887329, 593.6409302, -198.4790802, 578.8645630, -782.7531128, 792.1199951
4: -177.0714569, 558.4535522, -172.5252075, 544.8782349, -721.9497070, 730.9787598

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095952
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4098169, upper bound: 2585.4095935
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -861.4136963, 1666.8903809, -1381.2449951, 2683.0236816, -3544.4372559, 3048.1352539
1: -167.1999817, 214.2951508, -268.8007507, 345.6904907, -512.8905029, 483.0958862
2: -127.1888046, 283.2002869, -205.1590729, 443.7971191, -570.9859009, 488.3593750
3: -124.5308838, 370.2161865, -203.8620911, 592.0875244, -716.6184082, 574.0780640
4: -109.2905197, 355.7697449, -176.6902771, 557.5167236, -666.8071289, 532.4599609

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096278, upper bound: 2585.4096718
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096278, upper bound: 2585.4096718
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -861.4136963, 1666.8903809, -1417.1152344, 2763.6882324, -3625.1018066, 3084.0051270
1: -167.1999817, 214.2951508, -277.0158691, 355.3963013, -522.5962524, 491.3109741
2: -127.1888046, 283.2002869, -210.9283600, 456.5484924, -583.7373047, 494.1286621
3: -124.5308838, 370.2161865, -209.2034912, 608.9016113, -733.4324951, 579.4194336
4: -109.2905197, 355.7697449, -181.5829773, 573.3156128, -682.6060791, 537.3527222

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096278, upper bound: 2585.4096730
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096278, upper bound: 2585.4096730
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1407.2047119, 2747.6892090, -1381.2449951, 2683.0236816, -4090.2282715, 4127.0976562
1: -275.4772034, 353.1534119, -268.8007507, 345.6904907, -619.3106689, 620.8319702
2: -209.6806335, 453.6230164, -205.1590729, 443.7971191, -653.4777832, 658.7821045
3: -208.0089569, 605.2241821, -203.8620911, 592.0875244, -800.0964355, 809.0862427
4: -180.5376129, 569.6154175, -176.6902771, 557.5167236, -738.0543213, 746.3056030

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095918
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4095919
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1407.2047119, 2747.6892090, -1417.1152344, 2763.6882324, -4170.8925781, 4164.8041992
1: -275.4772034, 353.1534119, -277.0158691, 355.3963013, -629.9649048, 629.3131104
2: -209.6806335, 453.6230164, -210.9283600, 456.5484924, -666.2291260, 664.5513916
3: -208.0089569, 605.2241821, -209.2034912, 608.9016113, -816.9105835, 814.4276123
4: -180.5376129, 569.6154175, -181.5829773, 573.3156128, -753.8532104, 751.1983643

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096270, upper bound: 2585.4095931
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4096269, upper bound: 2585.4095912
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -891.0786133, 1714.1137695, -811.0416260, 1557.8208008, -2448.8994141, 2525.1552734
1: -171.9225922, 221.0413513, -156.2436066, 200.8871307, -372.8097229, 377.2849731
2: -131.3336182, 291.0024109, -119.1370163, 265.8622742, -397.1958923, 410.1394348
3: -129.6641388, 381.4725647, -116.8588333, 346.7650146, -476.4291382, 498.3313904
4: -112.9494247, 365.7009888, -102.3663788, 334.1200256, -447.0693665, 468.0673218

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095600, upper bound: 2585.4092662
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4095599, upper bound: 2585.4092661
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -891.0786133, 1714.1137695, -846.9887695, 1624.1540527, -2515.2326660, 2561.1025391
1: -171.9225922, 221.0413513, -162.7533875, 209.5159760, -381.4385376, 383.7947388
2: -131.3336182, 291.0024109, -124.3587952, 276.3965454, -407.7301636, 415.3611755
3: -129.6641388, 381.4725647, -122.1401901, 361.5578918, -491.2220154, 503.6127319
4: -112.9494247, 365.7009888, -106.8705215, 347.2706909, -460.2200928, 472.5715027

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4093302, upper bound: 2585.4092674
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4094305, upper bound: 2585.4092674
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -958.1411133, 1838.8131104, -777.6080322, 1497.7032471, -2455.8442383, 2616.4208984
1: -184.4638977, 237.1578522, -150.8075562, 193.0408173, -377.5046997, 387.9653015
2: -140.8832092, 313.5932617, -114.4368134, 254.9504242, -395.8336182, 428.0300903
3: -139.2979279, 409.5841064, -112.1500244, 332.7899170, -472.0878296, 521.7341309
4: -121.2188721, 393.9997864, -98.2738113, 320.4044800, -441.6233521, 492.2735901

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071030, upper bound: 2585.4078753
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2585.4071030, upper bound: 2585.4078753
time: 0.69 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.43 + 416.21 = 420.64 seconds
