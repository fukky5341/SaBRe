## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.6088244805
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.6422567, 2.6422567)
1: (-7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.9443550, 2.9443550)
2: (9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.9136095, 1.9136095)
3: (-4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648)
4: (-9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.7138886, 2.7138886)
5: (-13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.5247240, 2.5247238)
6: (-16.3375587, -12.7550831, -16.3375587, -12.7550831, -3.3848834, 3.3848829)
7: (-4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302)
8: (-6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.4180570, 2.4180570)
9: (-11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.5149889, 2.5149889)

## BASE Result
execution time: IAR + LP analysis = 14.38 + 32.68 = 47.06 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.94 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.7896461486816406
rel_dist={2: [-0.9337359263914102, 0.9337356752465436]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.56638765335083
rel_dist={2: [-0.6118841057361983, 0.6118843662850555]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.417548656463623
rel_dist={2: [-0.3841969223018733, 0.3841966429074404]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.4919681549072266
rel_dist={2: [-0.5006622952101019, 0.500662257900629]}

## Binary Search Result
Binary search time: 212.37 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3340.57 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328291, upper bound: 1.0193407
time: 6.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328289, upper bound: 1.0328299
time: 6.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.63
Output dim: 2, lower bound: -1.0328291, upper bound: 1.0193407
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.63
Output dim: 2, lower bound: -1.0328289, upper bound: 1.0328299

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.1174469, -10.4751902, -2.2002730, 2.2237353
1: -7.1278400, -4.1993065, -7.1292858, -4.1849308, -2.7251639, 2.7117529
2: 9.3696823, 11.2674522, 9.3677397, 11.2813492, -1.8613358, 1.8486943
3: -4.8695817, -2.7414377, -4.8719673, -2.7364025, -2.1331792, 2.1305296
4: -9.4359188, -6.7293978, -9.4387360, -6.7248473, -2.3674140, 2.3661094
5: -13.7910433, -11.1777372, -13.7978449, -11.1748800, -2.0125823, 2.0154185
6: -16.3353348, -12.7582769, -16.3375587, -12.7550831, -2.7731018, 2.7695084
7: -4.0544095, -1.3716025, -4.0563107, -1.3696804, -2.6847291, 2.6847081
8: -6.0347061, -3.6268172, -6.0375509, -3.6194940, -2.3039284, 2.3002896
9: -11.8219433, -9.3285122, -11.8428993, -9.3279104, -2.1376688, 2.1586006

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193392
time: 8.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193390
time: 10.16 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.1264296, -10.4119635, -13.1174278, -10.4751902, -2.2481976, 2.2455473
1: -7.1656313, -4.1722174, -7.1292849, -4.1849437, -2.7614431, 2.7552266
2: 9.3225374, 11.2848873, 9.3677435, 11.2813396, -1.8891649, 1.8689280
3: -4.8966494, -2.7341797, -4.8719654, -2.7364047, -2.1602447, 2.1377857
4: -9.4410944, -6.7103333, -9.4387350, -6.7248502, -2.3825421, 2.3874259
5: -13.8050451, -11.1572084, -13.7978411, -11.1748838, -2.0616117, 2.0351593
6: -16.3600388, -12.7376785, -16.3375607, -12.7550859, -2.7988429, 2.8003378
7: -4.0664396, -1.3625293, -4.0563097, -1.3696826, -2.6967571, 2.6937804
8: -6.0710683, -3.6127009, -6.0375490, -3.6195006, -2.3512392, 2.3149014
9: -11.8570776, -9.2782326, -11.8428783, -9.3279104, -2.1752696, 2.1830387

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0327177, upper bound: 1.0252793
time: 4.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328155, upper bound: 1.0328161
time: 6.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.60 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.60
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193392
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.60
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193390
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.60
Output dim: 2, lower bound: -1.0327177, upper bound: 1.0252793
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.60
Output dim: 2, lower bound: -1.0328155, upper bound: 1.0328161

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.0942993, -10.4755249, -2.1984572, 2.1984575
1: -7.1278400, -4.1993065, -7.1278400, -4.1993065, -2.7061205, 2.7061205
2: 9.3696823, 11.2674522, 9.3696823, 11.2674522, -1.8459644, 1.8459642
3: -4.8695817, -2.7414377, -4.8695817, -2.7414377, -2.1281440, 2.1281440
4: -9.4359188, -6.7293978, -9.4359188, -6.7293978, -2.3609567, 2.3609567
5: -13.7910433, -11.1777372, -13.7910433, -11.1777372, -2.0001531, 2.0001531
6: -16.3353348, -12.7582769, -16.3353348, -12.7582769, -2.7679224, 2.7679222
7: -4.0544095, -1.3716025, -4.0544095, -1.3716025, -2.6828070, 2.6828070
8: -6.0347061, -3.6268172, -6.0347061, -3.6268172, -2.2937784, 2.2937779
9: -11.8219433, -9.3285122, -11.8219433, -9.3285122, -2.1370726, 2.1370721

Time for backsubstitution: 12.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192373
time: 7.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
time: 8.64 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.1264086, -10.4119644, -2.2202611, 2.2346480
1: -7.1278400, -4.1993065, -7.1655025, -4.1722221, -2.7299871, 2.7422681
2: 9.3696823, 11.2674522, 9.3228989, 11.2848797, -1.8655391, 1.8734753
3: -4.8695817, -2.7414377, -4.8965921, -2.7341909, -2.1353908, 2.1551545
4: -9.4359188, -6.7293978, -9.4410906, -6.7108707, -2.3818431, 2.3667371
5: -13.7910433, -11.1777372, -13.8050432, -11.1578493, -2.0193682, 2.0140538
6: -16.3353348, -12.7582769, -16.3598595, -12.7376928, -2.7897968, 2.7933424
7: -4.0544095, -1.3716025, -4.0664282, -1.3625374, -2.6918721, 2.6948256
8: -6.0347061, -3.6268172, -6.0710535, -3.6127505, -2.3083706, 2.3310924
9: -11.8219433, -9.3285122, -11.8569212, -9.2782335, -2.1615014, 2.1737177

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192352
time: 7.39 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
time: 7.46 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.1257515, -10.4123716, -13.1144953, -10.4769735, -2.2454181, 2.2412381
1: -7.1642652, -4.1725984, -7.1234331, -4.1865883, -2.7589111, 2.7490602
2: 9.3254623, 11.2844257, 9.3803244, 11.2793350, -1.8811035, 1.8550456
3: -4.8961506, -2.7347360, -4.8698759, -2.7387710, -2.1573796, 2.1351399
4: -9.4396963, -6.7110872, -9.4327517, -6.7281437, -2.3776860, 2.3789454
5: -13.8047228, -11.1622353, -13.7964516, -11.1964674, -2.0395746, 2.0288076
6: -16.3596172, -12.7388830, -16.3357639, -12.7602425, -2.7907009, 2.7935171
7: -4.0606146, -1.3630564, -4.0313201, -1.3719838, -2.6886308, 2.6682637
8: -6.0700364, -3.6178970, -6.0330567, -3.6418195, -2.3275371, 2.3053322
9: -11.8541012, -9.2785397, -11.8301182, -9.3292332, -2.1711543, 2.1694195

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192355, upper bound: 1.0252775
time: 6.40 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192355, upper bound: 1.0252782
time: 7.01 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.1264277, -10.4119644, -13.1208897, -10.4712458, -2.2508559, 2.2490170
1: -7.1656303, -4.1722174, -7.1299486, -4.1781301, -2.7681341, 2.7556648
2: 9.3225441, 11.2848854, 9.3652363, 11.2971764, -1.8900795, 1.8685920
3: -4.8966494, -2.7341805, -4.8734827, -2.7353897, -2.1612597, 2.1393023
4: -9.4410915, -6.7103338, -9.4399691, -6.7172785, -2.3901954, 2.3878098
5: -13.8050442, -11.1572161, -13.8231506, -11.1743631, -2.0551786, 2.0483787
6: -16.3600388, -12.7376804, -16.3434505, -12.7524424, -2.8020372, 2.8034649
7: -4.0664301, -1.3625302, -4.0600090, -1.3428514, -2.7235787, 2.6974788
8: -6.0710649, -3.6127081, -6.0659242, -3.6171474, -2.3474679, 2.3438797
9: -11.8570766, -9.2782326, -11.8452578, -9.3175793, -2.1848688, 2.1831133

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193254, upper bound: 1.0328140
time: 8.14 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193254, upper bound: 1.0193247
time: 11.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 32.81 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192373
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192352
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0192355, upper bound: 1.0252775
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0192355, upper bound: 1.0252782
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0193254, upper bound: 1.0328140
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 2, lower bound: -1.0193254, upper bound: 1.0193247

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.0913610, -10.4773083, -13.0936127, -10.4759321, -2.1941583, 2.1956682
1: -7.1219940, -4.2009621, -7.1264777, -4.1996861, -2.6999578, 2.7035923
2: 9.3822632, 11.2654476, 9.3726139, 11.2669907, -1.8320832, 1.8394287
3: -4.8674974, -2.7438030, -4.8690891, -2.7419920, -2.1255054, 2.1252861
4: -9.4299364, -6.7326961, -9.4345217, -6.7301540, -2.3524756, 2.3561006
5: -13.7896509, -11.1993170, -13.7907248, -11.1827641, -1.9938035, 1.9781175
6: -16.3335457, -12.7634296, -16.3349190, -12.7594805, -2.7611041, 2.7597804
7: -4.0294218, -1.3739083, -4.0485830, -1.3721285, -2.6572933, 2.6746747
8: -6.0302114, -3.6491370, -6.0336771, -3.6320171, -2.2842159, 2.2700725
9: -11.8091946, -9.3298378, -11.8189669, -9.3288155, -2.1234562, 2.1329575

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0123389
time: 5.03 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0192270
time: 5.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715805, -13.0943031, -10.4755287, -2.2019160, 2.2011147
1: -7.1285033, -4.1924777, -7.1278396, -4.1993051, -2.7065616, 2.7128119
2: 9.3671894, 11.2832909, 9.3696890, 11.2674513, -1.8456354, 1.8604324
3: -4.8711004, -2.7404246, -4.8695817, -2.7414374, -2.1296630, 2.1291571
4: -9.4371519, -6.7218237, -9.4359159, -6.7293987, -2.3613391, 2.3686161
5: -13.8163528, -11.1772184, -13.7910433, -11.1777439, -2.0256715, 1.9937210
6: -16.3412170, -12.7556314, -16.3353348, -12.7582769, -2.7710428, 2.7711148
7: -4.0580997, -1.3447695, -4.0543995, -1.3716013, -2.6864984, 2.7096300
8: -6.0630856, -3.6244631, -6.0347042, -3.6268263, -2.3227701, 2.2900109
9: -11.8243227, -9.3181829, -11.8219395, -9.3285122, -2.1371160, 2.1466565

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0124307
time: 6.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0193152
time: 8.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.0913610, -10.4773083, -13.1257324, -10.4123735, -2.2159534, 2.2318685
1: -7.1219940, -4.2009621, -7.1641378, -4.1726046, -2.7238197, 2.7397337
2: 9.3822632, 11.2654476, 9.3258228, 11.2844191, -1.8516579, 1.8654160
3: -4.8674974, -2.7438030, -4.8960943, -2.7347460, -2.1327515, 2.1522913
4: -9.4299364, -6.7326961, -9.4396935, -6.7116232, -2.3733611, 2.3618813
5: -13.7896509, -11.1993170, -13.8047228, -11.1628771, -2.0130181, 1.9920187
6: -16.3335457, -12.7634296, -16.3594379, -12.7388992, -2.7829776, 2.7852020
7: -4.0294218, -1.3739083, -4.0606003, -1.3630638, -2.6663580, 2.6866920
8: -6.0302114, -3.6491370, -6.0700235, -3.6179447, -2.2988110, 2.3073878
9: -11.8091946, -9.3298378, -11.8539429, -9.2785387, -2.1478837, 2.1696010

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192233
time: 5.41 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0252632, upper bound: 1.0192233
time: 4.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715805, -13.1264124, -10.4119663, -2.2237260, 2.2373056
1: -7.1285033, -4.1924777, -7.1655016, -4.1722221, -2.7304268, 2.7489595
2: 9.3671894, 11.2832909, 9.3229036, 11.2848797, -1.8652101, 1.8743923
3: -4.8711004, -2.7404246, -4.8965921, -2.7341919, -2.1369085, 2.1561675
4: -9.4371519, -6.7218237, -9.4410877, -6.7108707, -2.3822260, 2.3743968
5: -13.8163528, -11.1772184, -13.8050442, -11.1578608, -2.0328708, 2.0076208
6: -16.3412170, -12.7556314, -16.3598576, -12.7376966, -2.7929173, 2.7965355
7: -4.0580997, -1.3447695, -4.0664186, -1.3625374, -2.6955624, 2.7216492
8: -6.0630856, -3.6244631, -6.0710516, -3.6127586, -2.3373632, 2.3273249
9: -11.8243227, -9.3181829, -11.8569193, -9.2782335, -2.1615758, 2.1833010

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0124268
time: 5.18 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0193109
time: 6.55 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.1257324, -10.4123735, -13.0913610, -10.4773083, -2.2318685, 2.2159536
1: -7.1641378, -4.1726046, -7.1219940, -4.2009621, -2.7397327, 2.7238188
2: 9.3258228, 11.2844191, 9.3822632, 11.2654476, -1.8654163, 1.8516576
3: -4.8960943, -2.7347460, -4.8674974, -2.7438030, -2.1522913, 2.1327515
4: -9.4396935, -6.7116232, -9.4299364, -6.7326961, -2.3618813, 2.3733613
5: -13.8047228, -11.1628771, -13.7896509, -11.1993170, -1.9920187, 2.0130184
6: -16.3594379, -12.7388992, -16.3335457, -12.7634296, -2.7852020, 2.7829776
7: -4.0606003, -1.3630638, -4.0294218, -1.3739083, -2.6866920, 2.6663580
8: -6.0700235, -3.6179447, -6.0302114, -3.6491370, -2.3073874, 2.2988110
9: -11.8539429, -9.2785387, -11.8091946, -9.3298378, -2.1696010, 2.1478839

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192214, upper bound: 1.0183994
time: 8.23 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192214, upper bound: 1.0252631
time: 9.06 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.1257496, -10.4123716, -13.1235275, -10.4137487, -2.2484241, 2.2469440
1: -7.1642942, -4.1725979, -7.1598024, -4.1738791, -2.7572422, 2.7536006
2: 9.3253765, 11.2844267, 9.3350086, 11.2828846, -1.8744025, 1.8670485
3: -4.8961620, -2.7347319, -4.8945503, -2.7365460, -2.1596160, 2.1598184
4: -9.4396963, -6.7109618, -9.4351110, -6.7134953, -2.3845239, 2.3809023
5: -13.8047218, -11.1620865, -13.8036537, -11.1786404, -2.0418105, 2.0548906
6: -16.3596592, -12.7388792, -16.3582745, -12.7428226, -2.8093195, 2.8106618
7: -4.0606165, -1.3630548, -4.0414495, -1.3648298, -2.6957867, 2.6783948
8: -6.0700397, -3.6178827, -6.0666089, -3.6349907, -2.3347769, 2.3489170
9: -11.8541355, -9.2785397, -11.8443556, -9.2795610, -2.1800621, 2.1705570

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0123348, upper bound: 1.0252663
time: 4.78 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192209, upper bound: 1.0252663
time: 14.59 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.1264124, -10.4119663, -13.0977631, -10.4715805, -2.2373056, 2.2237258
1: -7.1655016, -4.1722221, -7.1285033, -4.1924777, -2.7489595, 2.7304268
2: 9.3229036, 11.2848797, 9.3671894, 11.2832909, -1.8743923, 1.8652098
3: -4.8965921, -2.7341919, -4.8711004, -2.7404246, -2.1561675, 2.1369085
4: -9.4410877, -6.7108707, -9.4371519, -6.7218237, -2.3743968, 2.3822260
5: -13.8050442, -11.1578608, -13.8163528, -11.1772184, -2.0076208, 2.0328708
6: -16.3598576, -12.7376966, -16.3412170, -12.7556314, -2.7965355, 2.7929173
7: -4.0664186, -1.3625374, -4.0580997, -1.3447695, -2.7216492, 2.6955624
8: -6.0710516, -3.6127586, -6.0630856, -3.6244631, -2.3273249, 2.3373632
9: -11.8569193, -9.2782335, -11.8243227, -9.3181829, -2.1833010, 2.1615758

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124266, upper bound: 1.0328001
time: 8.47 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193108, upper bound: 1.0328022
time: 5.10 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.1264305, -10.4119644, -13.1298771, -10.4080219, -2.2538593, 2.2546678
1: -7.1656585, -4.1722164, -7.1663308, -4.1653929, -2.7664733, 2.7602172
2: 9.3224621, 11.2848864, 9.3199577, 11.3007202, -1.8944252, 1.8806436
3: -4.8966608, -2.7341771, -4.8981905, -2.7331629, -2.1634979, 2.1640134
4: -9.4410915, -6.7102094, -9.4423294, -6.7026329, -2.3970280, 2.3897724
5: -13.8050442, -11.1570702, -13.8303518, -11.1565361, -2.0574744, 2.0658660
6: -16.3600826, -12.7376785, -16.3659859, -12.7350512, -2.8206949, 2.8206336
7: -4.0664315, -1.3625276, -4.0701437, -1.3356965, -2.7307351, 2.7076161
8: -6.0710683, -3.6126986, -6.0994430, -3.6103745, -2.3546658, 2.3668904
9: -11.8571091, -9.2782335, -11.8594837, -9.2678967, -2.1937895, 2.1842284

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124266, upper bound: 1.0328001
time: 9.43 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193108, upper bound: 1.0328010
time: 7.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.30 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0123389
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0192270
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0124307
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0193152
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192233
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0252632, upper bound: 1.0192233
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0124268
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0193109
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0192214, upper bound: 1.0183994
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0192214, upper bound: 1.0252631
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0123348, upper bound: 1.0252663
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0192209, upper bound: 1.0252663
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0124266, upper bound: 1.0328001
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0193108, upper bound: 1.0328022
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0124266, upper bound: 1.0328001
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 2, lower bound: -1.0193108, upper bound: 1.0328010

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.0913610, -10.4773083, -13.0925255, -10.4890633, -2.1796012, 2.1949618
1: -7.1219940, -4.2009621, -7.1254659, -4.2055492, -2.6890378, 2.6981692
2: 9.3822632, 11.2654476, 9.3804674, 11.2657003, -1.8291869, 1.8307576
3: -4.8674974, -2.7438030, -4.8682957, -2.7444711, -2.1230264, 2.1244926
4: -9.4299364, -6.7326961, -9.4323273, -6.7373934, -2.3446755, 2.3491478
5: -13.7896509, -11.1993170, -13.7902412, -11.1927404, -1.9836135, 1.9776785
6: -16.3335457, -12.7634296, -16.3343010, -12.7631636, -2.7544775, 2.7592680
7: -4.0294218, -1.3739083, -4.0445247, -1.3787723, -2.6506495, 2.6706164
8: -6.0302114, -3.6491370, -6.0321326, -3.6412392, -2.2750082, 2.2682614
9: -11.8091946, -9.3298378, -11.8162031, -9.3304996, -2.1218383, 2.1295476

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0123390
time: 6.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0123386
time: 7.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.0913591, -10.4773159, -13.1479492, -10.4703770, -2.2053902, 2.2285399
1: -7.1219931, -4.2009649, -7.1501160, -4.1876540, -2.7286434, 2.7204227
2: 9.3822680, 11.2654467, 9.3578920, 11.2931919, -1.8480015, 1.8620577
3: -4.8674970, -2.7438049, -4.8854609, -2.7368197, -2.1306772, 2.1416559
4: -9.4299345, -6.7327027, -9.4548302, -6.7250795, -2.3609648, 2.3730812
5: -13.7896500, -11.1993246, -13.8296738, -11.1793470, -1.9979177, 2.0113800
6: -16.3335438, -12.7634315, -16.3617477, -12.7303581, -2.8054724, 2.7876656
7: -4.0294199, -1.3739123, -4.0798874, -1.3688598, -2.6605601, 2.7059751
8: -6.0302114, -3.6491418, -6.0767307, -3.6272840, -2.2909722, 2.3128719
9: -11.8091946, -9.3298397, -11.8420448, -9.3206024, -2.1325281, 2.1591706

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0117917
time: 5.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0192270
time: 5.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715805, -13.0932150, -10.4886551, -2.1873608, 2.2004085
1: -7.1285033, -4.1924777, -7.1268258, -4.2051682, -2.6956429, 2.7073865
2: 9.3671894, 11.2832909, 9.3775463, 11.2661619, -1.8427382, 1.8517661
3: -4.8711004, -2.7404246, -4.8687849, -2.7439201, -2.1271803, 2.1283603
4: -9.4371519, -6.7218237, -9.4337177, -6.7366381, -2.3535438, 2.3616586
5: -13.8163528, -11.1772184, -13.7905636, -11.1877251, -2.0154815, 1.9932814
6: -16.3412170, -12.7556314, -16.3347187, -12.7619610, -2.7644157, 2.7706041
7: -4.0580997, -1.3447695, -4.0503359, -1.3782458, -2.6798539, 2.7055664
8: -6.0630856, -3.6244631, -6.0331645, -3.6360478, -2.3135624, 2.2882032
9: -11.8243227, -9.3181829, -11.8191776, -9.3301983, -2.1354985, 2.1432495

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0124328
time: 4.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0124308
time: 6.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715891, -13.1486273, -10.4699688, -2.2131443, 2.2327828
1: -7.1285033, -4.1924829, -7.1514769, -4.1872692, -2.7352467, 2.7296443
2: 9.3671942, 11.2832899, 9.3549767, 11.2936497, -1.8605318, 1.8789525
3: -4.8710995, -2.7404258, -4.8859525, -2.7362623, -2.1348372, 2.1455266
4: -9.4371510, -6.7218313, -9.4562149, -6.7243204, -2.3698125, 2.3855727
5: -13.8163509, -11.1772270, -13.8299971, -11.1743279, -2.0297861, 2.0270443
6: -16.3412170, -12.7556314, -16.3621635, -12.7291574, -2.8154101, 2.7977240
7: -4.0580978, -1.3447738, -4.0856819, -1.3683290, -2.6897688, 2.7409081
8: -6.0630841, -3.6244683, -6.0777607, -3.6220970, -2.3295269, 2.3328934
9: -11.8243208, -9.3181829, -11.8449965, -9.3202982, -2.1461873, 2.1728506

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192253, upper bound: 1.0117914
time: 7.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0192257, upper bound: 1.0052038
time: 7.59 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -13.0902729, -10.4904385, -13.1257324, -10.4123735, -2.2152591, 2.2173054
1: -7.1209850, -4.2068224, -7.1641378, -4.1726046, -2.7184010, 2.7288132
2: 9.3901119, 11.2641497, 9.3258228, 11.2844191, -1.8429852, 1.8625379
3: -4.8667021, -2.7462735, -4.8960943, -2.7347460, -2.1319561, 2.1498208
4: -9.4277525, -6.7399364, -9.4396935, -6.7116232, -2.3664265, 2.3540943
5: -13.7891693, -11.2092896, -13.8047228, -11.1628771, -2.0125799, 1.9818306
6: -16.3329277, -12.7671061, -16.3594379, -12.7388992, -2.7824678, 2.7785699
7: -4.0253687, -1.3805480, -4.0606003, -1.3630638, -2.6623049, 2.6800523
8: -6.0286655, -3.6583600, -6.0700235, -3.6179447, -2.2969923, 2.2981815
9: -11.8064260, -9.3315220, -11.8539429, -9.2785387, -2.1444700, 2.1679809

Time for backsubstitution: 12.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0123348
time: 10.85 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192233
time: 8.90 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -13.1457214, -10.4717579, -13.1257324, -10.4123821, -2.2309403, 2.2431095
1: -7.1456375, -4.1889367, -7.1641369, -4.1726084, -2.7406416, 2.7567167
2: 9.3675175, 11.2916565, 9.3258266, 11.2844172, -1.8742781, 1.8679373
3: -4.8838682, -2.7386446, -4.8960934, -2.7347479, -2.1491203, 2.1574488
4: -9.4502773, -6.7276335, -9.4396906, -6.7116294, -2.3904252, 2.3704247
5: -13.8285990, -11.1959038, -13.8047209, -11.1628866, -2.0303402, 1.9961302
6: -16.3603706, -12.7343006, -16.3594398, -12.7389021, -2.8108635, 2.8240416
7: -4.0607901, -1.3706532, -4.0605984, -1.3630679, -2.6977222, 2.6899452
8: -6.0733500, -3.6443949, -6.0700226, -3.6179509, -2.3391671, 2.3141422
9: -11.8323441, -9.3216324, -11.8539410, -9.2785397, -2.1682777, 2.1786685

Time for backsubstitution: 12.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0252632, upper bound: 1.0117878
time: 5.79 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0252632, upper bound: 1.0192233
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715805, -13.1252708, -10.4250689, -2.2091379, 2.2365575
1: -7.1285033, -4.1924777, -7.1644793, -4.1780958, -2.7194862, 2.7435465
2: 9.3671894, 11.2832909, 9.3307133, 11.2835789, -1.8623466, 1.8656895
3: -4.8711004, -2.7404246, -4.8957853, -2.7366688, -2.1344316, 2.1553607
4: -9.4371519, -6.7218237, -9.4388752, -6.7180929, -2.3744249, 2.3674426
5: -13.8163528, -11.1772184, -13.8045330, -11.1677923, -2.0226731, 2.0071545
6: -16.3412170, -12.7556314, -16.3592224, -12.7413273, -2.7861896, 2.7960038
7: -4.0580997, -1.3447695, -4.0623317, -1.3691797, -2.6889200, 2.7175622
8: -6.0630856, -3.6244631, -6.0695062, -3.6219740, -2.3281651, 2.3255553
9: -11.8243227, -9.3181829, -11.8541250, -9.2798882, -2.1599340, 2.1798668

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259242, upper bound: 1.0124268
time: 6.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0124264
time: 6.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715891, -13.1807623, -10.4064226, -2.2313619, 2.2688656
1: -7.1285033, -4.1924829, -7.1892414, -4.1606073, -2.7592363, 2.7503183
2: 9.3671942, 11.2832899, 9.3081360, 11.3110704, -1.8803811, 1.8927000
3: -4.8710995, -2.7404258, -4.9131193, -2.7291157, -2.1419837, 2.1726935
4: -9.4371510, -6.7218313, -9.4613628, -6.7057924, -2.3907757, 2.3913233
5: -13.8163509, -11.1772270, -13.8439074, -11.1543369, -2.0370173, 2.0407653
6: -16.3412170, -12.7556314, -16.3868065, -12.7093010, -2.8370748, 2.8105319
7: -4.0580978, -1.3447738, -4.0982475, -1.3592629, -2.6988349, 2.7534738
8: -6.0630841, -3.6244683, -6.1142511, -3.6081519, -2.3440752, 2.3526292
9: -11.8243208, -9.3181829, -11.8796883, -9.2700777, -2.1697927, 2.2094350

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0327026, upper bound: 1.0117852
time: 6.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0327030, upper bound: 1.0117872
time: 7.47 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -13.1257324, -10.4123735, -13.0902729, -10.4904385, -2.2173052, 2.2152591
1: -7.1641378, -4.1726046, -7.1209850, -4.2068224, -2.7288132, 2.7184010
2: 9.3258228, 11.2844191, 9.3901119, 11.2641497, -1.8625379, 1.8429849
3: -4.8960943, -2.7347460, -4.8667021, -2.7462735, -2.1498208, 2.1319561
4: -9.4396935, -6.7116232, -9.4277525, -6.7399364, -2.3540945, 2.3664265
5: -13.8047228, -11.1628771, -13.7891693, -11.2092896, -1.9818306, 2.0125799
6: -16.3594379, -12.7388992, -16.3329277, -12.7671061, -2.7785702, 2.7824678
7: -4.0606003, -1.3630638, -4.0253687, -1.3805480, -2.6800523, 2.6623049
8: -6.0700235, -3.6179447, -6.0286655, -3.6583600, -2.2981815, 2.2969923
9: -11.8539429, -9.2785387, -11.8064260, -9.3315220, -2.1679811, 2.1444700

Time for backsubstitution: 12.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0123348, upper bound: 1.0184017
time: 4.99 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0123348, upper bound: 1.0184017
time: 4.85 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -13.1257324, -10.4123821, -13.1457214, -10.4717579, -2.2431095, 2.2309403
1: -7.1641369, -4.1726084, -7.1456375, -4.1889367, -2.7567165, 2.7406421
2: 9.3258266, 11.2844172, 9.3675175, 11.2916565, -1.8679373, 1.8742776
3: -4.8960934, -2.7347479, -4.8838682, -2.7386446, -2.1574488, 2.1491203
4: -9.4396906, -6.7116294, -9.4502773, -6.7276335, -2.3704247, 2.3904254
5: -13.8047209, -11.1628866, -13.8285990, -11.1959038, -1.9961300, 2.0303402
6: -16.3594398, -12.7389021, -16.3603706, -12.7343006, -2.8240416, 2.8108635
7: -4.0605984, -1.3630679, -4.0607901, -1.3706532, -2.6899452, 2.6977222
8: -6.0700226, -3.6179509, -6.0733500, -3.6443949, -2.3141422, 2.3391671
9: -11.8539410, -9.2785397, -11.8323441, -9.3216324, -2.1786687, 2.1682780

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117858, upper bound: 1.0252630
time: 5.05 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117858, upper bound: 1.0252627
time: 8.78 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1246128, -10.4254780, -13.1235275, -10.4137487, -2.2477016, 2.2323663
1: -7.1632733, -4.1784716, -7.1598024, -4.1738791, -2.7518272, 2.7427311
2: 9.3331871, 11.2831268, 9.3350086, 11.2828846, -1.8657150, 1.8641891
3: -4.8953571, -2.7372065, -4.8945503, -2.7365460, -2.1588111, 2.1573439
4: -9.4374876, -6.7181864, -9.4351110, -6.7134953, -2.3775778, 2.3730869
5: -13.8042126, -11.1720181, -13.8036537, -11.1786404, -2.0413799, 2.0446820
6: -16.3590260, -12.7425098, -16.3582745, -12.7428226, -2.8087893, 2.8040180
7: -4.0565319, -1.3696947, -4.0414495, -1.3648298, -2.6917021, 2.6717548
8: -6.0684929, -3.6270971, -6.0666089, -3.6349907, -2.3330059, 2.3397198
9: -11.8513422, -9.2801895, -11.8443556, -9.2795610, -2.1766250, 2.1689301

Time for backsubstitution: 12.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0128295, upper bound: 1.0184014
time: 7.77 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0128295, upper bound: 1.0252645
time: 7.80 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1801014, -10.4068308, -13.1235275, -10.4137564, -2.2728765, 2.2581315
1: -7.1880360, -4.1609836, -7.1598001, -4.1738834, -2.7741203, 2.7827721
2: 9.3106022, 11.3106203, 9.3350134, 11.2828846, -1.8971844, 1.8821867
3: -4.9126797, -2.7296591, -4.8945498, -2.7365479, -2.1761317, 2.1648908
4: -9.4599800, -6.7058892, -9.4351101, -6.7135005, -2.4015632, 2.3893452
5: -13.8435850, -11.1585636, -13.8036528, -11.1786489, -2.0501177, 2.0589867
6: -16.3866043, -12.7104836, -16.3582726, -12.7428274, -2.8256648, 2.8494437
7: -4.0924602, -1.3597860, -4.0414453, -1.3648367, -2.7276235, 2.6816592
8: -6.1132650, -3.6132741, -6.0666080, -3.6349955, -2.3528533, 2.3556285
9: -11.8769207, -9.2703838, -11.8443527, -9.2795620, -2.2062154, 2.1797066

Time for backsubstitution: 12.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0122791, upper bound: 1.0252654
time: 5.66 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0122791, upper bound: 1.0252643
time: 8.57 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1252708, -10.4250689, -13.0977631, -10.4715805, -2.2365575, 2.2091379
1: -7.1644793, -4.1780958, -7.1285033, -4.1924777, -2.7435465, 2.7194862
2: 9.3307133, 11.2835789, 9.3671894, 11.2832909, -1.8656895, 1.8623469
3: -4.8957853, -2.7366688, -4.8711004, -2.7404246, -2.1553607, 2.1344316
4: -9.4388752, -6.7180929, -9.4371519, -6.7218237, -2.3674426, 2.3744252
5: -13.8045330, -11.1677923, -13.8163528, -11.1772184, -2.0071545, 2.0226731
6: -16.3592224, -12.7413273, -16.3412170, -12.7556314, -2.7960033, 2.7861896
7: -4.0623317, -1.3691797, -4.0580997, -1.3447695, -2.7175622, 2.6889200
8: -6.0695062, -3.6219740, -6.0630856, -3.6244631, -2.3255558, 2.3281651
9: -11.8541250, -9.2798882, -11.8243227, -9.3181829, -2.1798668, 2.1599345

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124266, upper bound: 1.0259241
time: 11.16 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124266, upper bound: 1.0328024
time: 5.01 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1807623, -10.4064226, -13.0977631, -10.4715891, -2.2688656, 2.2313619
1: -7.1892414, -4.1606073, -7.1285033, -4.1924829, -2.7503183, 2.7592359
2: 9.3081360, 11.3110704, 9.3671942, 11.2832899, -1.8927004, 1.8803813
3: -4.9131193, -2.7291157, -4.8710995, -2.7404258, -2.1726935, 2.1419837
4: -9.4613628, -6.7057924, -9.4371510, -6.7218313, -2.3913231, 2.3907759
5: -13.8439074, -11.1543369, -13.8163509, -11.1772270, -2.0407653, 2.0370173
6: -16.3868065, -12.7093010, -16.3412170, -12.7556314, -2.8105319, 2.8370750
7: -4.0982475, -1.3592629, -4.0580978, -1.3447738, -2.7534738, 2.6988349
8: -6.1142511, -3.6081519, -6.0630841, -3.6244683, -2.3526297, 2.3440757
9: -11.8796883, -9.2700777, -11.8243208, -9.3181829, -2.2094347, 2.1697927

Time for backsubstitution: 12.40 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8640661239624023
rel_dist={2: [-1.032862430954955, 1.0328621621306109]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7217719, upper bound: 0.7140570
time: 7.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218423, upper bound: 0.7218413
time: 7.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.14
Output dim: 2, lower bound: -0.7217719, upper bound: 0.7140570
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.14
Output dim: 2, lower bound: -0.7218423, upper bound: 0.7218413

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.1099072, -10.4752979, -1.8820953, 1.8979359
1: -7.1278400, -4.1993065, -7.1288347, -4.1897826, -2.4206748, 2.4115996
2: 9.3696823, 11.2674522, 9.3683577, 11.2768211, -1.6330605, 1.6244926
3: -4.8695817, -2.7414377, -4.8712025, -2.7380447, -1.9958887, 1.9927945
4: -9.4359188, -6.7293978, -9.4378138, -6.7262793, -2.0603886, 2.0595269
5: -13.7910433, -11.1777372, -13.7955914, -11.1757803, -1.7103853, 1.7123325
6: -16.3353348, -12.7582769, -16.3368568, -12.7561407, -2.4053078, 2.4028795
7: -4.0544095, -1.3716025, -4.0556979, -1.3702984, -2.5370598, 2.5373449
8: -6.0347061, -3.6268172, -6.0366421, -3.6218872, -2.0892525, 2.0868006
9: -11.8219433, -9.3285122, -11.8360348, -9.3281040, -1.8230171, 1.8370819

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7212330, upper bound: 0.7090760
time: 6.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7217635, upper bound: 0.7140507
time: 5.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.1264191, -10.4119635, -13.1174116, -10.4751873, -1.9197423, 1.9251745
1: -7.1655688, -4.1722183, -7.1292839, -4.1849518, -2.4604125, 2.4499235
2: 9.3227129, 11.2848835, 9.3677435, 11.2813320, -1.6627250, 1.6396708
3: -4.8966208, -2.7341847, -4.8719635, -2.7364068, -2.0272636, 2.0091772
4: -9.4410906, -6.7105942, -9.4387350, -6.7248535, -2.0755267, 2.0823078
5: -13.8050451, -11.1575212, -13.7978382, -11.1748848, -1.7502933, 1.7367716
6: -16.3599529, -12.7376881, -16.3375587, -12.7550869, -2.4325652, 2.4329839
7: -4.0664349, -1.3625331, -4.0563097, -1.3696833, -2.5536332, 2.5486994
8: -6.0710602, -3.6127253, -6.0375462, -3.6195035, -2.1367116, 2.1035180
9: -11.8570042, -9.2782335, -11.8428669, -9.3279095, -1.8535662, 1.8649716

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7213037, upper bound: 0.7168645
time: 7.19 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218339, upper bound: 0.7218322
time: 9.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 30.81 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.81
Output dim: 2, lower bound: -0.7212330, upper bound: 0.7090760
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.81
Output dim: 2, lower bound: -0.7217635, upper bound: 0.7140507
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.81
Output dim: 2, lower bound: -0.7213037, upper bound: 0.7168645
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.81
Output dim: 2, lower bound: -0.7218339, upper bound: 0.7218322

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.0928650, -10.4763803, -13.1069736, -10.4770813, -1.8783321, 1.8931625
1: -7.1250038, -4.2000990, -7.1229849, -4.1914272, -2.4166622, 2.4051423
2: 9.3757820, 11.2664871, 9.3809357, 11.2748184, -1.6232343, 1.6097274
3: -4.8685598, -2.7425900, -4.8691158, -2.7404115, -1.9897575, 1.9871111
4: -9.4330139, -6.7309794, -9.4318314, -6.7295728, -2.0535722, 2.0502844
5: -13.7903738, -11.1882000, -13.7941990, -11.1973639, -1.6880183, 1.7005086
6: -16.3344650, -12.7607803, -16.3350620, -12.7612953, -2.3958507, 2.3943207
7: -4.0422869, -1.3727078, -4.0307083, -1.3726006, -2.5161014, 2.5054970
8: -6.0325508, -3.6376381, -6.0321507, -3.6442046, -2.0644779, 2.0715055
9: -11.8157520, -9.3291473, -11.8232784, -9.3294277, -1.8155303, 1.8232164

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135158, upper bound: 0.7090736
time: 11.74 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135158, upper bound: 0.7090734
time: 6.63 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.0943031, -10.4755249, -13.1133718, -10.4713545, -1.8847523, 1.9011879
1: -7.1278381, -4.1993060, -7.1294994, -4.1829576, -2.4273624, 2.4118080
2: 9.3696928, 11.2674522, 9.3658562, 11.2926598, -1.6434798, 1.6206896
3: -4.8695812, -2.7414370, -4.8727212, -2.7370298, -1.9948335, 1.9957819
4: -9.4359159, -6.7293987, -9.4390488, -6.7187047, -2.0680428, 2.0596752
5: -13.7910461, -11.1777534, -13.8209000, -11.1752625, -1.6986279, 1.7308463
6: -16.3353329, -12.7582788, -16.3427410, -12.7534981, -2.4070249, 2.4060028
7: -4.0543909, -1.3716025, -4.0593939, -1.3434663, -2.5588541, 2.5297022
8: -6.0347033, -3.6268330, -6.0650191, -3.6195335, -2.0807219, 2.1130157
9: -11.8219376, -9.3285141, -11.8384113, -9.3177738, -1.8326094, 1.8356068

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140515, upper bound: 0.7140507
time: 5.72 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140515, upper bound: 0.7140481
time: 10.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.1250057, -10.4128180, -13.1144781, -10.4769745, -1.9159975, 1.9203898
1: -7.1627254, -4.1730175, -7.1234336, -4.1865983, -2.4557400, 2.4434686
2: 9.3287983, 11.2839203, 9.3803244, 11.2793283, -1.6510699, 1.6249053
3: -4.8955889, -2.7353377, -4.8698730, -2.7387729, -2.0211191, 2.0034924
4: -9.4381847, -6.7121706, -9.4327507, -6.7281466, -2.0687094, 2.0730639
5: -13.8043747, -11.1679840, -13.7964458, -11.1964693, -1.7278969, 1.7249460
6: -16.3590736, -12.7401886, -16.3357658, -12.7602434, -2.4231138, 2.4244189
7: -4.0543089, -1.3636382, -4.0313168, -1.3719859, -2.5326676, 2.5168514
8: -6.0689058, -3.6235390, -6.0330553, -3.6418228, -2.1119142, 2.0882316
9: -11.8508034, -9.2788706, -11.8301048, -9.3292322, -1.8460751, 1.8511033

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7212744, upper bound: 0.7122964
time: 6.79 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7212993, upper bound: 0.7168623
time: 4.57 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.1264172, -10.4119663, -13.1208735, -10.4712448, -1.9223969, 1.9284325
1: -7.1655645, -4.1722202, -7.1299477, -4.1781378, -2.4624643, 2.4501328
2: 9.3227215, 11.2848797, 9.3652363, 11.2971697, -1.6636400, 1.6358643
3: -4.8966212, -2.7341857, -4.8734827, -2.7353916, -2.0262089, 2.0121632
4: -9.4410887, -6.7105966, -9.4399672, -6.7172823, -2.0831776, 2.0824561
5: -13.8050451, -11.1575365, -13.8231487, -11.1743660, -1.7385488, 1.7428834
6: -16.3599510, -12.7376919, -16.3434467, -12.7524452, -2.4334335, 2.4361086
7: -4.0664167, -1.3625336, -4.0600085, -1.3428524, -2.5722070, 2.5410576
8: -6.0710588, -3.6127410, -6.0659218, -3.6171513, -2.1282158, 2.1298428
9: -11.8569956, -9.2782345, -11.8452435, -9.3175812, -1.8631642, 1.8635190

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218050, upper bound: 0.7172653
time: 4.95 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218295, upper bound: 0.7218274
time: 9.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.17 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7135158, upper bound: 0.7090736
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7135158, upper bound: 0.7090734
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7140515, upper bound: 0.7140507
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7140515, upper bound: 0.7140481
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7212744, upper bound: 0.7122964
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7212993, upper bound: 0.7168623
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7218050, upper bound: 0.7172653
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 29.17
Output dim: 2, lower bound: -0.7218295, upper bound: 0.7218274

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -13.0928650, -10.4763803, -13.0913610, -10.4773083, -1.8771083, 1.8761001
1: -7.1250038, -4.2000990, -7.1219940, -4.2009621, -2.4037952, 2.4013538
2: 9.3757820, 11.2664871, 9.3822632, 11.2654476, -1.6128812, 1.6079412
3: -4.8685598, -2.7425900, -4.8674974, -2.7438030, -1.9843035, 1.9847512
4: -9.4330139, -6.7309794, -9.4299364, -6.7326961, -2.0492325, 2.0468068
5: -13.7903738, -11.1882000, -13.7896509, -11.1993170, -1.6796613, 1.6902051
6: -16.3344650, -12.7607803, -16.3335457, -12.7634296, -2.3923497, 2.3932471
7: -4.0422869, -1.3727078, -4.0294218, -1.3739083, -2.5144610, 2.5035720
8: -6.0325508, -3.6376381, -6.0302114, -3.6491370, -2.0576353, 2.0671272
9: -11.8157520, -9.3291473, -11.8091946, -9.3298378, -1.8151240, 1.8087471

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7090471
time: 9.50 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135109, upper bound: 0.7090691
time: 9.13 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -13.0928650, -10.4763803, -13.1234970, -10.4137478, -1.8959322, 1.9122283
1: -7.1250038, -4.2000990, -7.1595597, -4.1738892, -2.4276314, 2.4348655
2: 9.3757820, 11.2664871, 9.3356934, 11.2828703, -1.6324511, 1.6320169
3: -4.8685598, -2.7425900, -4.8944435, -2.7365670, -1.9924889, 2.0133996
4: -9.4330139, -6.7309794, -9.4351072, -6.7145181, -2.0698252, 2.0525875
5: -13.7903738, -11.1882000, -13.8036528, -11.1798630, -1.6985188, 1.7041113
6: -16.3344650, -12.7607803, -16.3579292, -12.7428541, -2.4142075, 2.4184606
7: -4.0422869, -1.3727078, -4.0414267, -1.3648431, -2.5248938, 2.5167956
8: -6.0325508, -3.6376381, -6.0665846, -3.6350865, -2.0722260, 2.1044393
9: -11.8157520, -9.3291473, -11.8440571, -9.2795601, -1.8347194, 1.8452625

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7090471
time: 4.89 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135109, upper bound: 0.7090716
time: 5.57 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.0943031, -10.4755249, -13.0977631, -10.4715805, -1.8835292, 1.8841195
1: -7.1278381, -4.1993060, -7.1285033, -4.1924777, -2.4144983, 2.4080195
2: 9.3696928, 11.2674522, 9.3671894, 11.2832909, -1.6341450, 1.6189067
3: -4.8695812, -2.7414370, -4.8711004, -2.7404246, -1.9893775, 1.9934235
4: -9.4359159, -6.7293987, -9.4371519, -6.7218237, -2.0637088, 2.0561976
5: -13.7910461, -11.1777534, -13.8163528, -11.1772184, -1.6902709, 1.7216159
6: -16.3353329, -12.7582788, -16.3412170, -12.7556314, -2.4035220, 2.4049251
7: -4.0543909, -1.3716025, -4.0580997, -1.3447695, -2.5572166, 2.5277753
8: -6.0347033, -3.6268330, -6.0630856, -3.6244631, -2.0738850, 2.1092725
9: -11.8219376, -9.3285141, -11.8243227, -9.3181829, -1.8321934, 1.8211360

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094820, upper bound: 0.7140206
time: 12.49 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140466, upper bound: 0.7140443
time: 6.56 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.0943031, -10.4755249, -13.1298456, -10.4080210, -1.9011929, 1.9201503
1: -7.1278381, -4.1993060, -7.1660891, -4.1654024, -2.4383554, 2.4415379
2: 9.3696928, 11.2674522, 9.3206444, 11.3007069, -1.6483862, 1.6419508
3: -4.8695812, -2.7414370, -4.8980865, -2.7331846, -1.9975667, 2.0218983
4: -9.4359159, -6.7293987, -9.4423256, -6.7036552, -2.0842948, 2.0619760
5: -13.7910461, -11.1777534, -13.8303518, -11.1577616, -1.7091327, 1.7338098
6: -16.3353329, -12.7582788, -16.3656406, -12.7350817, -2.4253912, 2.4283786
7: -4.0543909, -1.3716025, -4.0701227, -1.3357098, -2.5646420, 2.5409932
8: -6.0347033, -3.6268330, -6.0994172, -3.6104693, -2.0884275, 2.1291506
9: -11.8219376, -9.3285141, -11.8591881, -9.2678967, -1.8458090, 1.8576627

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094820, upper bound: 0.7140227
time: 8.37 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140466, upper bound: 0.7140439
time: 7.38 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.1247196, -10.4161568, -13.1133881, -10.4901066, -1.9012504, 1.9152527
1: -7.1624665, -4.1745291, -7.1224246, -4.1925802, -2.4432340, 2.4352856
2: 9.3307877, 11.2835922, 9.3881702, 11.2780313, -1.6455140, 1.6154995
3: -4.8953857, -2.7359710, -4.8690853, -2.7412481, -2.0163794, 2.0004911
4: -9.4376287, -6.7140098, -9.4305639, -6.7353888, -2.0591297, 2.0641413
5: -13.8042488, -11.1705122, -13.7959614, -11.2064342, -1.7175775, 1.7219269
6: -16.3589134, -12.7411184, -16.3351498, -12.7639475, -2.4163632, 2.4222269
7: -4.0532789, -1.3653278, -4.0272675, -1.3786278, -2.5244102, 2.5093355
8: -6.0685182, -3.6258845, -6.0315161, -3.6510453, -2.1022506, 2.0840411
9: -11.8500986, -9.2792912, -11.8273325, -9.3309212, -1.8435900, 1.8472109

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7134846, upper bound: 0.7122252
time: 8.42 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7134846, upper bound: 0.7122950
time: 8.17 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.1250086, -10.4128342, -13.1687851, -10.4713936, -1.9204288, 1.9354241
1: -7.1627264, -4.1730232, -7.1471000, -4.1747632, -2.4635136, 2.4584227
2: 9.3288059, 11.2839184, 9.3655415, 11.3055239, -1.6535852, 1.6432345
3: -4.8955870, -2.7353423, -4.8862619, -2.7336230, -2.0279131, 2.0201459
4: -9.4381838, -6.7121811, -9.4530764, -6.7230663, -2.0737910, 2.0888023
5: -13.8043737, -11.1679974, -13.8353558, -11.1930046, -1.7287633, 1.7344060
6: -16.3590736, -12.7401943, -16.3626175, -12.7311840, -2.4550552, 2.4418166
7: -4.0543051, -1.3636436, -4.0626292, -1.3687191, -2.5346317, 2.5478039
8: -6.0689039, -3.6235471, -6.0762210, -3.6370931, -2.1152396, 2.1212206
9: -11.8508034, -9.2788725, -11.8531590, -9.3210049, -1.8545432, 1.8714156

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135090, upper bound: 0.7167890
time: 8.09 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135090, upper bound: 0.7168604
time: 6.01 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.1261311, -10.4153042, -13.1197729, -10.4843721, -1.9076631, 1.9232826
1: -7.1653070, -4.1737318, -7.1289310, -4.1841183, -2.4499545, 2.4419422
2: 9.3247128, 11.2845554, 9.3730850, 11.2958851, -1.6580820, 1.6264520
3: -4.8964176, -2.7348213, -4.8726892, -2.7378817, -2.0214548, 2.0091658
4: -9.4405279, -6.7124372, -9.4377422, -6.7245097, -2.0736113, 2.0735030
5: -13.8049154, -11.1600647, -13.8226604, -11.1843386, -1.7282274, 1.7393841
6: -16.3597889, -12.7386236, -16.3428268, -12.7561522, -2.4267030, 2.4339118
7: -4.0653853, -1.3642268, -4.0559144, -1.3494914, -2.5639176, 2.5335255
8: -6.0706692, -3.6150866, -6.0643911, -3.6263742, -2.1185541, 2.1251912
9: -11.8562908, -9.2786589, -11.8424559, -9.3192635, -1.8606734, 1.8596213

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140211, upper bound: 0.7171950
time: 5.56 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140211, upper bound: 0.7172661
time: 5.34 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.1264162, -10.4119787, -13.1751757, -10.4656601, -1.9268241, 1.9433727
1: -7.1655655, -4.1722269, -7.1536160, -4.1662779, -2.4702439, 2.4650991
2: 9.3227301, 11.2848797, 9.3505297, 11.3233614, -1.6661582, 1.6542532
3: -4.8966198, -2.7341890, -4.8898702, -2.7302442, -2.0332947, 2.0288301
4: -9.4410858, -6.7106066, -9.4602375, -6.7121835, -2.0882506, 2.0981913
5: -13.8050461, -11.1575508, -13.8620567, -11.1709089, -1.7394128, 1.7513534
6: -16.3599510, -12.7376957, -16.3702927, -12.7234459, -2.4645419, 2.4511569
7: -4.0664129, -1.3625417, -4.0912294, -1.3395813, -2.5741329, 2.5719595
8: -6.0710549, -3.6127481, -6.1089940, -3.6124725, -2.1315136, 2.1438153
9: -11.8569956, -9.2782354, -11.8682270, -9.3093491, -1.8716452, 1.8838031

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140447, upper bound: 0.7217571
time: 7.05 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140447, upper bound: 0.7218274
time: 7.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.18 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7090471
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7135109, upper bound: 0.7090691
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7090471
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7135109, upper bound: 0.7090716
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7094820, upper bound: 0.7140206
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7140466, upper bound: 0.7140443
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7094820, upper bound: 0.7140227
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7140466, upper bound: 0.7140439
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7134846, upper bound: 0.7122252
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7134846, upper bound: 0.7122950
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7135090, upper bound: 0.7167890
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7135090, upper bound: 0.7168604
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7140211, upper bound: 0.7171950
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7140211, upper bound: 0.7172661
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7140447, upper bound: 0.7217571
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 2, lower bound: -0.7140447, upper bound: 0.7218274

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.0917778, -10.4895096, -13.0910816, -10.4806509, -1.8726978, 1.8613627
1: -7.1239920, -4.2059631, -7.1217389, -4.2024674, -2.3956051, 2.3890548
2: 9.3836346, 11.2651958, 9.3842630, 11.2651196, -1.6034713, 1.6028404
3: -4.8677659, -2.7450662, -4.8672981, -2.7444370, -1.9813085, 1.9799981
4: -9.4308205, -6.7382183, -9.4293842, -6.7345381, -2.0402994, 2.0372362
5: -13.7898912, -11.1981735, -13.7895317, -11.2018566, -1.6766286, 1.6799078
6: -16.3338509, -12.7644615, -16.3333874, -12.7643709, -2.3901572, 2.3864889
7: -4.0382290, -1.3793485, -4.0283980, -1.3756001, -2.5069470, 2.4953213
8: -6.0310073, -3.6468611, -6.0298252, -3.6514821, -2.0534773, 2.0574627
9: -11.8129864, -9.3308334, -11.8084974, -9.3302689, -1.8112974, 1.8062627

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7045055
time: 6.81 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7090474
time: 7.07 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1472139, -10.4708252, -13.0913582, -10.4773235, -1.9071765, 1.8805580
1: -7.1486449, -4.1880703, -7.1219945, -4.2009678, -2.4206181, 2.4253507
2: 9.3610516, 11.2926903, 9.3822708, 11.2654457, -1.6312628, 1.6208153
3: -4.8849335, -2.7374225, -4.8674965, -2.7438073, -2.0010166, 1.9968648
4: -9.4533319, -6.7259111, -9.4299355, -6.7327080, -2.0662384, 2.0519013
5: -13.8293238, -11.1847858, -13.7896500, -11.1993275, -1.7056725, 1.6910551
6: -16.3612938, -12.7316542, -16.3335438, -12.7634335, -2.4163213, 2.4351826
7: -4.0736127, -1.3694422, -4.0294166, -1.3739150, -2.5453520, 2.5055313
8: -6.0756149, -3.6329036, -6.0302129, -3.6491446, -2.0943069, 2.0703974
9: -11.8388500, -9.3209410, -11.8091936, -9.3298388, -1.8413572, 1.8172095

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090715, upper bound: 0.7090715
time: 6.65 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090715, upper bound: 0.7090722
time: 5.46 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.0917778, -10.4895096, -13.1232052, -10.4170876, -1.8907886, 1.8974802
1: -7.1239920, -4.2059631, -7.1593013, -4.1753988, -2.4194260, 2.4223332
2: 9.3836346, 11.2651958, 9.3376808, 11.2825432, -1.6230497, 1.6264594
3: -4.8677659, -2.7450662, -4.8942413, -2.7371981, -1.9894948, 2.0086560
4: -9.4308205, -6.7382183, -9.4345493, -6.7163568, -2.0608916, 2.0430160
5: -13.7898912, -11.1981735, -13.8035269, -11.1823940, -1.6954975, 1.6938066
6: -16.3338509, -12.7644615, -16.3577690, -12.7437811, -2.4119902, 2.4116971
7: -4.0382290, -1.3793485, -4.0403957, -1.3665369, -2.5173807, 2.5085444
8: -6.0310073, -3.6468611, -6.0661941, -3.6374321, -2.0680704, 2.0947833
9: -11.8129864, -9.3308334, -11.8433495, -9.2799873, -1.8308353, 1.8427715

Time for backsubstitution: 13.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7166629, upper bound: 0.7045030
time: 7.71 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7166629, upper bound: 0.7090446
time: 6.68 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1472139, -10.4708252, -13.1234970, -10.4137640, -1.9109035, 1.9166868
1: -7.1486449, -4.1880703, -7.1595602, -4.1738949, -2.4444566, 2.4426930
2: 9.3610516, 11.2926903, 9.3357029, 11.2828703, -1.6500871, 1.6345406
3: -4.8849335, -2.7374225, -4.8944426, -2.7365694, -2.0092025, 2.0201049
4: -9.4533319, -6.7259111, -9.4351034, -6.7145267, -2.0852137, 2.0576801
5: -13.8293238, -11.1847858, -13.8036537, -11.1798763, -1.7113867, 1.7049608
6: -16.3612938, -12.7316542, -16.3579292, -12.7428608, -2.4382045, 2.4501913
7: -4.0736127, -1.3694422, -4.0414219, -1.3648524, -2.5557861, 2.5187550
8: -6.0756149, -3.6329036, -6.0665817, -3.6350946, -2.1066623, 2.1077099
9: -11.8388500, -9.3209410, -11.8440571, -9.2795620, -1.8551183, 1.8537257

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7167890, upper bound: 0.7090695
time: 8.65 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7167899, upper bound: 0.7090716
time: 8.24 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -13.0932112, -10.4886580, -13.0974846, -10.4749250, -1.8791208, 1.8693833
1: -7.1268229, -4.2051702, -7.1282449, -4.1939859, -2.4063082, 2.3957176
2: 9.3775482, 11.2661619, 9.3691902, 11.2829676, -1.6246817, 1.6138000
3: -4.8687859, -2.7439206, -4.8709021, -2.7410612, -1.9863806, 1.9886694
4: -9.4337177, -6.7366371, -9.4365892, -6.7236662, -2.0547686, 2.0466232
5: -13.7905636, -11.1877298, -13.8162327, -11.1797600, -1.6872382, 1.7113018
6: -16.3347168, -12.7619619, -16.3410568, -12.7565727, -2.4013300, 2.3981664
7: -4.0503278, -1.3782473, -4.0570655, -1.3464608, -2.5493526, 2.5194979
8: -6.0331621, -3.6360550, -6.0626988, -3.6268106, -2.0697322, 2.0996106
9: -11.8191757, -9.3301964, -11.8236160, -9.3186131, -1.8283741, 1.8186510

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094821, upper bound: 0.7094815
time: 5.99 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094821, upper bound: 0.7140250
time: 7.34 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1486292, -10.4699678, -13.0977612, -10.4715939, -1.9124103, 1.8885722
1: -7.1514740, -4.1872725, -7.1285019, -4.1924849, -2.4313269, 2.4320168
2: 9.3549786, 11.2936516, 9.3671961, 11.2832899, -1.6484127, 1.6307604
3: -4.8859529, -2.7362618, -4.8710999, -2.7404275, -2.0060940, 2.0055251
4: -9.4562130, -6.7243214, -9.4371519, -6.7218351, -2.0806642, 2.0612574
5: -13.8299961, -11.1743345, -13.8163548, -11.1772299, -1.7163239, 1.7224883
6: -16.3621674, -12.7291603, -16.3412132, -12.7556334, -2.4258344, 2.4468160
7: -4.0856752, -1.3683286, -4.0580950, -1.3447757, -2.5731864, 2.5297403
8: -6.0777597, -3.6221037, -6.0630836, -3.6244698, -2.1106114, 2.1126261
9: -11.8449917, -9.3202982, -11.8243189, -9.3181820, -1.8583875, 1.8295996

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4586

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7123310, upper bound: 0.7139939
time: 7.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140449, upper bound: 0.7140472
time: 5.52 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.0932112, -10.4886580, -13.1295547, -10.4113607, -1.8960488, 1.9054027
1: -7.1268229, -4.2051702, -7.1658306, -4.1669116, -2.4301481, 2.4290051
2: 9.3775482, 11.2661619, 9.3226299, 11.3003807, -1.6389318, 1.6363850
3: -4.8687859, -2.7439206, -4.8978825, -2.7338190, -1.9945674, 2.0171170
4: -9.4337177, -6.7366371, -9.4417572, -6.7054920, -2.0753541, 2.0524035
5: -13.7905636, -11.1877298, -13.8302250, -11.1602898, -1.7061119, 1.7234892
6: -16.3347168, -12.7619619, -16.3654785, -12.7360106, -2.4231658, 2.4216383
7: -4.0503278, -1.3782473, -4.0690813, -1.3374009, -2.5567789, 2.5327139
8: -6.0331621, -3.6360550, -6.0990314, -3.6128168, -2.0842767, 2.1194859
9: -11.8191757, -9.3301964, -11.8584747, -9.2683182, -1.8419275, 1.8551710

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7171934, upper bound: 0.7094813
time: 5.23 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7171934, upper bound: 0.7140227
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1486292, -10.4699678, -13.1298447, -10.4080353, -1.9161372, 1.9246030
1: -7.1514740, -4.1872725, -7.1660876, -4.1654096, -2.4551826, 2.4493713
2: 9.3549786, 11.2936516, 9.3206501, 11.3007040, -1.6626534, 1.6444767
3: -4.8859529, -2.7362618, -4.8980856, -2.7331877, -2.0142837, 2.0272846
4: -9.4562130, -6.7243214, -9.4423218, -6.7036657, -2.0955446, 2.0670352
5: -13.8299961, -11.1743345, -13.8303509, -11.1577721, -1.7220378, 1.7346816
6: -16.3621674, -12.7291603, -16.3656387, -12.7350874, -2.4477098, 2.4595404
7: -4.0856752, -1.3683286, -4.0701180, -1.3357167, -2.5806131, 2.5429583
8: -6.0777597, -3.6221037, -6.0994139, -3.6104774, -2.1229239, 2.1325045
9: -11.8449917, -9.3202982, -11.8591843, -9.2678995, -1.8662014, 1.8661273

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7216330, upper bound: 0.7123286
time: 6.61 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7217561, upper bound: 0.7140422
time: 6.22 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -13.1246996, -10.4161568, -13.0902729, -10.4904385, -1.8984604, 1.8899635
1: -7.1623173, -4.1745324, -7.1209850, -4.2068224, -2.4241014, 2.4170070
2: 9.3312140, 11.2835817, 9.3901119, 11.2641497, -1.6297708, 1.6181049
3: -4.8953190, -2.7359838, -4.8667021, -2.7462735, -2.0082202, 1.9899430
4: -9.4376259, -6.7146487, -9.4277525, -6.7399364, -2.0454540, 2.0584826
5: -13.8042469, -11.1712713, -13.7891693, -11.2092896, -1.6832614, 1.7060413
6: -16.3587017, -12.7411385, -16.3329277, -12.7671061, -2.4107914, 2.4128921
7: -4.0532646, -1.3653364, -4.0253687, -1.3805480, -2.5194077, 2.5065093
8: -6.0685019, -3.6259437, -6.0286655, -3.6583600, -2.0852890, 2.0775504
9: -11.8499117, -9.2792931, -11.8064260, -9.3315220, -1.8491545, 1.8256736

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122251
time: 6.68 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122250
time: 5.98 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -13.1247292, -10.4161568, -13.1223879, -10.4268541, -1.9044251, 1.9137728
1: -7.1625595, -4.1745243, -7.1587815, -4.1797519, -2.4382048, 2.4398389
2: 9.3305283, 11.2835960, 9.3428116, 11.2815762, -1.6367979, 1.6274974
3: -4.8954248, -2.7359624, -4.8937449, -2.7390106, -2.0084844, 2.0106945
4: -9.4376287, -6.7136269, -9.4329128, -6.7207203, -2.0659633, 2.0642109
5: -13.8042469, -11.1700497, -13.8031425, -11.1885681, -1.7200828, 1.7343125
6: -16.3590450, -12.7411098, -16.3576355, -12.7464504, -2.4338789, 2.4393544
7: -4.0532875, -1.3653231, -4.0373697, -1.3714695, -2.5303726, 2.5202456
8: -6.0685272, -3.6258488, -6.0650578, -3.6442022, -2.1095996, 2.1210616
9: -11.8502102, -9.2792931, -11.8415546, -9.2812157, -1.8525712, 1.8448272

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122965
time: 6.09 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122960
time: 8.57 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.1249886, -10.4128342, -13.1457214, -10.4717579, -1.9176757, 1.9100914
1: -7.1625762, -4.1730313, -7.1456375, -4.1889367, -2.4444511, 2.4420276
2: 9.3292332, 11.2839079, 9.3675175, 11.2916565, -1.6378472, 1.6458826
3: -4.8955212, -2.7353544, -4.8838682, -2.7386446, -2.0197923, 2.0096502
4: -9.4381790, -6.7128181, -9.4502773, -6.7276335, -2.0601420, 2.0834446
5: -13.8043718, -11.1687593, -13.8285990, -11.1959038, -1.6944132, 1.7188271
6: -16.3588619, -12.7402134, -16.3603706, -12.7343006, -2.4496720, 2.4387140
7: -4.0542908, -1.3636513, -4.0607901, -1.3706532, -2.5296168, 2.5449576
8: -6.0688887, -3.6236072, -6.0733500, -3.6443949, -2.0982113, 2.1131148
9: -11.8506165, -9.2788715, -11.8323441, -9.3216324, -1.8601048, 1.8499639

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7167885
time: 8.79 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7167892
time: 4.81 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -13.1250172, -10.4128323, -13.1778708, -10.4082127, -1.9236097, 1.9411583
1: -7.1628180, -4.1730213, -7.1835475, -4.1622734, -2.4716406, 2.4629760
2: 9.3285475, 11.2839241, 9.3202095, 11.3090849, -1.6529541, 1.6554432
3: -4.8956285, -2.7353334, -4.9110675, -2.7314868, -2.0253921, 2.0303802
4: -9.4381847, -6.7117939, -9.4554281, -6.7084341, -2.0806336, 2.0902472
5: -13.8043728, -11.1675339, -13.8425131, -11.1751194, -1.7312114, 1.7461472
6: -16.3592072, -12.7401848, -16.3852100, -12.7144232, -2.4730959, 2.4532971
7: -4.0543137, -1.3636384, -4.0733585, -1.3615761, -2.5405784, 2.5589070
8: -6.0689144, -3.6235094, -6.1099124, -3.6303697, -2.1225505, 2.1396410
9: -11.8509150, -9.2788715, -11.8672085, -9.2714167, -1.8636081, 1.8749099

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7168597
time: 9.06 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7168600
time: 8.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 32.22 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7045055
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7089450, upper bound: 0.7090474
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7090715, upper bound: 0.7090715
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7090715, upper bound: 0.7090722
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7166629, upper bound: 0.7045030
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7166629, upper bound: 0.7090446
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7167890, upper bound: 0.7090695
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7167899, upper bound: 0.7090716
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7094821, upper bound: 0.7094815
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7094821, upper bound: 0.7140250
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7123310, upper bound: 0.7139939
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7140449, upper bound: 0.7140472
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7171934, upper bound: 0.7094813
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7171934, upper bound: 0.7140227
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7216330, upper bound: 0.7123286
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7217561, upper bound: 0.7140422
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122251
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122250
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122965
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7122960
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7167885
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7167892
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7168597
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.22
Output dim: 2, lower bound: -0.7090695, upper bound: 0.7168600
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 32.22
Output dim: 2, lower bound: -0.7140211, upper bound: 0.7171950
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.22
Output dim: 2, lower bound: -0.7140211, upper bound: 0.7172661
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.22
Output dim: 2, lower bound: -0.7140447, upper bound: 0.7217571
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.22
Output dim: 2, lower bound: -0.7140447, upper bound: 0.7218274
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.6408071517944336
rel_dist={2: [-0.7218570837425542, 0.7218564741351408]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117883, upper bound: 0.6060459
time: 7.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118731, upper bound: 0.6118744
time: 6.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.20
Output dim: 2, lower bound: -0.6117883, upper bound: 0.6060459
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.20
Output dim: 2, lower bound: -0.6118731, upper bound: 0.6118744

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.1065817, -10.4753466, -1.7759726, 1.7884445
1: -7.1278400, -4.1993065, -7.1286306, -4.1918344, -2.3185024, 2.3113518
2: 9.3696823, 11.2674522, 9.3686333, 11.2748222, -1.5564303, 1.5496728
3: -4.8695817, -2.7414377, -4.8708601, -2.7387688, -1.9142661, 1.9118214
4: -9.4359188, -6.7293978, -9.4374084, -6.7269254, -1.9578218, 1.9571505
5: -13.7910433, -11.1777372, -13.7946081, -11.1761875, -1.6092186, 1.6107650
6: -16.3353348, -12.7582769, -16.3365383, -12.7566013, -2.2825251, 2.2806125
7: -4.0544095, -1.3716025, -4.0554237, -1.3705735, -2.4592524, 2.4594755
8: -6.0347061, -3.6268172, -6.0362363, -3.6229424, -2.0173373, 2.0154009
9: -11.8219433, -9.3285122, -11.8330135, -9.3281918, -1.7181113, 1.7291594

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6111854, upper bound: 0.6021157
time: 9.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117819, upper bound: 0.6060391
time: 5.84 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.1264162, -10.4119644, -13.1174068, -10.4751921, -1.8102477, 1.8183835
1: -7.1655455, -4.1722198, -7.1292839, -4.1849556, -2.3584208, 2.3481569
2: 9.3227787, 11.2848835, 9.3677444, 11.2813282, -1.5872386, 1.5632510
3: -4.8966112, -2.7341871, -4.8719616, -2.7364085, -1.9466476, 1.9276290
4: -9.4410906, -6.7106919, -9.4387321, -6.7248545, -1.9731865, 1.9805927
5: -13.8050442, -11.1576366, -13.7978354, -11.1748838, -1.6463671, 1.6372995
6: -16.3599167, -12.7376900, -16.3375587, -12.7550888, -2.3102825, 2.3105321
7: -4.0664330, -1.3625340, -4.0563078, -1.3696835, -2.4760466, 2.4712348
8: -6.0710573, -3.6127348, -6.0375452, -3.6195049, -2.0637965, 2.0330553
9: -11.8569736, -9.2782335, -11.8428583, -9.3279095, -1.7463284, 1.7589490

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112706, upper bound: 0.6079467
time: 8.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118666, upper bound: 0.6118675
time: 13.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 36.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.47
Output dim: 2, lower bound: -0.6111854, upper bound: 0.6021157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.47
Output dim: 2, lower bound: -0.6117819, upper bound: 0.6060391
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.47
Output dim: 2, lower bound: -0.6112706, upper bound: 0.6079467
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.47
Output dim: 2, lower bound: -0.6118666, upper bound: 0.6118675

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.0925570, -10.4765692, -13.1036434, -10.4771299, -1.7718039, 1.7834733
1: -7.1243925, -4.2002754, -7.1227822, -4.1934886, -2.3138723, 2.3047729
2: 9.3771000, 11.2662783, 9.3812132, 11.2728186, -1.5452337, 1.5345409
3: -4.8683414, -2.7428384, -4.8687735, -2.7411346, -1.9077640, 1.9056740
4: -9.4323864, -6.7313275, -9.4314280, -6.7302184, -1.9501901, 1.9475889
5: -13.7902279, -11.1904602, -13.7932129, -11.1977692, -1.5867195, 1.5966632
6: -16.3342819, -12.7613201, -16.3347473, -12.7617588, -2.2725234, 2.2713275
7: -4.0396671, -1.3729482, -4.0304375, -1.3728771, -2.4353247, 2.4268699
8: -6.0320787, -3.6399789, -6.0317440, -3.6452608, -1.9921145, 1.9977221
9: -11.8144131, -9.3292866, -11.8202591, -9.3295145, -1.7092206, 1.7151897

Time for backsubstitution: 13.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054442, upper bound: 0.6021157
time: 9.28 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054442, upper bound: 0.6021181
time: 10.20 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.0942984, -10.4755249, -13.1100435, -10.4714022, -1.7786293, 1.7916255
1: -7.1278377, -4.1993070, -7.1292953, -4.1850100, -2.3251920, 2.3114839
2: 9.3696957, 11.2674522, 9.3661337, 11.2906628, -1.5657585, 1.5447135
3: -4.8695803, -2.7414379, -4.8723803, -2.7377548, -1.9132109, 1.9146156
4: -9.4359140, -6.7294002, -9.4386435, -6.7193513, -1.9654779, 1.9572206
5: -13.7910452, -11.1777563, -13.8199158, -11.1756668, -1.5956869, 1.6267757
6: -16.3353348, -12.7582817, -16.3424225, -12.7539577, -2.2837515, 2.2837353
7: -4.0543876, -1.3716021, -4.0591202, -1.3437414, -2.4791231, 2.4496851
8: -6.0347047, -3.6268358, -6.0646124, -3.6205883, -2.0072212, 2.0396407
9: -11.8219366, -9.3285141, -11.8353930, -9.3178616, -1.7277012, 1.7271767

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060415, upper bound: 0.6060395
time: 7.73 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060415, upper bound: 0.6060383
time: 5.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.1246996, -10.4130058, -13.1144724, -10.4769735, -1.8061028, 1.8133991
1: -7.1620889, -4.1731939, -7.1234322, -4.1866007, -2.3531895, 2.3415785
2: 9.3301792, 11.2837076, 9.3803253, 11.2793245, -1.5743771, 1.5481191
3: -4.8953567, -2.7355902, -4.8698730, -2.7387750, -1.9398732, 1.9214811
4: -9.4375601, -6.7126126, -9.4327507, -6.7281466, -1.9655528, 1.9710307
5: -13.8042278, -11.1703615, -13.7964439, -11.1964684, -1.6238365, 1.6220806
6: -16.3588524, -12.7407341, -16.3357639, -12.7602434, -2.2999039, 2.3012381
7: -4.0516872, -1.3638797, -4.0313177, -1.3719859, -2.4521122, 2.4386287
8: -6.0684314, -3.6258864, -6.0330553, -3.6418228, -2.0385523, 2.0153899
9: -11.8494387, -9.2790079, -11.8301001, -9.3292351, -1.7374330, 1.7449753

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112304, upper bound: 0.6044941
time: 8.75 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112674, upper bound: 0.6079406
time: 7.23 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.1264162, -10.4119663, -13.1208658, -10.4712467, -1.8129051, 1.8215699
1: -7.1655412, -4.1722217, -7.1299477, -4.1781425, -2.3604748, 2.3482900
2: 9.3227882, 11.2848816, 9.3652372, 11.2971668, -1.5881541, 1.5582869
3: -4.8966093, -2.7341876, -4.8734803, -2.7353921, -1.9456019, 1.9304218
4: -9.4410858, -6.7106938, -9.4399672, -6.7172813, -1.9808378, 1.9806633
5: -13.8050451, -11.1576538, -13.8231478, -11.1743660, -1.6328406, 1.6410465
6: -16.3599167, -12.7376938, -16.3434448, -12.7524462, -2.3094323, 2.3136582
7: -4.0664120, -1.3625345, -4.0600071, -1.3428531, -2.4926972, 2.4614453
8: -6.0710535, -3.6127534, -6.0659227, -3.6171532, -2.0537062, 2.0573387
9: -11.8569660, -9.2782345, -11.8452368, -9.3175812, -1.7559252, 1.7569866

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118289, upper bound: 0.6084235
time: 8.25 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118634, upper bound: 0.6118638
time: 32.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 55.27 seconds
IS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6054442, upper bound: 0.6021157
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6054442, upper bound: 0.6021181
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6060415, upper bound: 0.6060395
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6060415, upper bound: 0.6060383
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6112304, upper bound: 0.6044941
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6112674, upper bound: 0.6079406
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6118289, upper bound: 0.6084235
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 55.27
Output dim: 2, lower bound: -0.6118634, upper bound: 0.6118638

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.1242275, -10.4184418, -13.1133833, -10.4901066, -1.7912414, 1.8057654
1: -7.1616688, -4.1756473, -7.1224246, -4.1925840, -2.3397660, 2.3316541
2: 9.3334160, 11.2831707, 9.3881721, 11.2780266, -1.5673258, 1.5382562
3: -4.8950262, -2.7366185, -4.8690834, -2.7412488, -1.9347792, 1.9177957
4: -9.4366493, -6.7156091, -9.4305639, -6.7353888, -1.9548640, 1.9608593
5: -13.8040199, -11.1744785, -13.7959595, -11.2064352, -1.6134493, 1.6168436
6: -16.3585930, -12.7422457, -16.3351479, -12.7639484, -2.2930903, 2.2979865
7: -4.0500069, -1.3666348, -4.0272689, -1.3786287, -2.4429455, 2.4300289
8: -6.0677967, -3.6297054, -6.0315137, -3.6510468, -2.0286002, 2.0097280
9: -11.8482819, -9.2796993, -11.8273277, -9.3309221, -1.7343960, 1.7408066

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6078211, upper bound: 0.6044934
time: 9.27 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6078211, upper bound: 0.6044936
time: 8.44 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.1246986, -10.4130211, -13.1687784, -10.4713945, -1.8082767, 1.8284335
1: -7.1620884, -4.1732025, -7.1471000, -4.1747661, -2.3593981, 2.3539836
2: 9.3301868, 11.2837086, 9.3655405, 11.3055210, -1.5768921, 1.5650344
3: -4.8953562, -2.7355931, -4.8862610, -2.7336240, -1.9447000, 1.9381328
4: -9.4375572, -6.7126250, -9.4530764, -6.7230663, -1.9694986, 1.9854269
5: -13.8042307, -11.1703758, -13.8353539, -11.1930046, -1.6236098, 1.6305625
6: -16.3588524, -12.7407379, -16.3626175, -12.7311859, -2.3302240, 2.3172774
7: -4.0516810, -1.3638890, -4.0626278, -1.3687198, -2.4533386, 2.4695807
8: -6.0684304, -3.6258955, -6.0762215, -3.6370940, -2.0407138, 2.0466008
9: -11.8494377, -9.2790117, -11.8531523, -9.3210049, -1.7456973, 1.7652884

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054391, upper bound: 0.6078545
time: 8.54 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054391, upper bound: 0.6079412
time: 11.00 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.1259422, -10.4174013, -13.1197643, -10.4843712, -1.7980537, 1.8139253
1: -7.1651211, -4.1746769, -7.1289282, -4.1841211, -2.3470449, 2.3383560
2: 9.3260279, 11.2843456, 9.3730869, 11.2958822, -1.5811036, 1.5484154
3: -4.8962784, -2.7352195, -4.8726888, -2.7378821, -1.9404933, 1.9267397
4: -9.4401760, -6.7136889, -9.4377394, -6.7245102, -1.9701591, 1.9704633
5: -13.8048363, -11.1617756, -13.8226585, -11.1843386, -1.6224504, 1.6358066
6: -16.3596535, -12.7392063, -16.3428249, -12.7561522, -2.3026125, 2.3103986
7: -4.0647287, -1.3652906, -4.0559130, -1.3494906, -2.4834790, 2.4528279
8: -6.0704217, -3.6165714, -6.0643902, -3.6263776, -2.0437589, 2.0511091
9: -11.8558159, -9.2789221, -11.8424482, -9.3192616, -1.7528870, 1.7528124

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 4586
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6084260
time: 6.10 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6084234
time: 6.18 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.1264133, -10.4119787, -13.1751671, -10.4656630, -1.8150713, 1.8365105
1: -7.1655407, -4.1722288, -7.1536169, -4.1662812, -2.3666866, 2.3607030
2: 9.3227997, 11.2848778, 9.3505306, 11.3233595, -1.5906720, 1.5752618
3: -4.8966093, -2.7341921, -4.8898702, -2.7302456, -1.9504247, 1.9469023
4: -9.4410830, -6.7107048, -9.4602375, -6.7121854, -1.9847732, 1.9950547
5: -13.8050423, -11.1576710, -13.8620558, -11.1709080, -1.6326115, 1.6495161
6: -16.3599167, -12.7376986, -16.3702927, -12.7234459, -2.3397245, 2.3272736
7: -4.0664072, -1.3625438, -4.0912290, -1.3395805, -2.4938836, 2.4922247
8: -6.0710521, -3.6127625, -6.1089931, -3.6124735, -2.0558405, 2.0713091
9: -11.8569651, -9.2782354, -11.8682175, -9.3093510, -1.7642040, 1.7772703

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060364, upper bound: 0.6117766
time: 7.80 seconds

## Relational analysis of IS_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117195, upper bound: 0.6104829
time: 9.84 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118622, upper bound: 0.6118599
time: 19.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 78.95 seconds
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6078211, upper bound: 0.6044934
IS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6078211, upper bound: 0.6044936
IS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6054391, upper bound: 0.6078545
IS_A2_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6054391, upper bound: 0.6079412
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6084260
IS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6084234
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6117195, upper bound: 0.6104829
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 78.95
Output dim: 2, lower bound: -0.6118622, upper bound: 0.6118599

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -13.1252136, -10.4172688, -13.1662035, -10.4759331, -1.8036938, 1.8207688
1: -7.1642470, -4.1769581, -7.1454787, -4.1754990, -2.3555121, 2.3462465
2: 9.3272305, 11.2836494, 9.3598232, 11.3160591, -1.5773506, 1.5642509
3: -4.8955832, -2.7368622, -4.8837128, -2.7352836, -1.9442286, 1.9384389
4: -9.4387121, -6.7145014, -9.4523382, -6.7197332, -1.9723091, 1.9770741
5: -13.8032331, -11.1593151, -13.8559284, -11.1745243, -1.6272135, 1.6420587
6: -16.3589458, -12.7423849, -16.3610210, -12.7355051, -2.3263628, 2.3135557
7: -4.0620766, -1.3630364, -4.0816226, -1.3413997, -2.4849982, 2.4794545
8: -6.0698080, -3.6186247, -6.0991654, -3.6238961, -2.0425363, 2.0525012
9: -11.8501816, -9.2783861, -11.8529463, -9.3145752, -1.7520158, 1.7619736

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6058927, upper bound: 0.6103988
time: 8.37 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6058927, upper bound: 0.6104865
time: 5.35 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.1264153, -10.4119835, -13.1751661, -10.4656715, -1.8062739, 1.8315084
1: -7.1655378, -4.1722336, -7.1536150, -4.1662922, -2.3601682, 2.3555722
2: 9.3227997, 11.2848797, 9.3505335, 11.3233566, -1.5869765, 1.5679855
3: -4.8966074, -2.7341950, -4.8898678, -2.7302499, -1.9480021, 1.9434931
4: -9.4410820, -6.7107096, -9.4602337, -6.7121916, -1.9796467, 1.9917347
5: -13.8050423, -11.1576748, -13.8620539, -11.1709080, -1.6303074, 1.6480513
6: -16.3599167, -12.7377033, -16.3702927, -12.7234516, -2.3319204, 2.3224261
7: -4.0664024, -1.3625441, -4.0912228, -1.3395817, -2.4941382, 2.4914441
8: -6.0710506, -3.6127682, -6.1089916, -3.6124868, -2.0460606, 2.0658796
9: -11.8569584, -9.2782354, -11.8682079, -9.3093500, -1.7641981, 1.7684932

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4586

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060352, upper bound: 0.6117749
time: 9.54 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060352, upper bound: 0.6118632
time: 7.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.77 seconds
IS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 31.77
Output dim: 2, lower bound: -0.6058927, upper bound: 0.6103988
IS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 31.77
Output dim: 2, lower bound: -0.6058927, upper bound: 0.6104865
IS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 31.77
Output dim: 2, lower bound: -0.6060352, upper bound: 0.6117749
IS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 31.77
Output dim: 2, lower bound: -0.6060352, upper bound: 0.6118632

## BFS IS instance: IS_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -13.1251945, -10.4172697, -13.1431446, -10.4762869, -1.8026221, 1.7954257
1: -7.1640873, -4.1769657, -7.1440210, -4.1896644, -2.3364253, 2.3347363
2: 9.3276854, 11.2836390, 9.3618107, 11.3021851, -1.5615835, 1.5657389
3: -4.8955135, -2.7368767, -4.8813162, -2.7403073, -1.9360988, 1.9291935
4: -9.4387083, -6.7151785, -9.4495335, -6.7242875, -1.9593720, 1.9717178
5: -13.8032312, -11.1601248, -13.8491621, -11.1774254, -1.5973334, 1.6264783
6: -16.3587189, -12.7424068, -16.3588028, -12.7386227, -2.3209734, 2.3108368
7: -4.0620632, -1.3630433, -4.0797834, -1.3433249, -2.4822092, 2.4765892
8: -6.0697913, -3.6186891, -6.0963173, -3.6311989, -2.0279522, 2.0435023
9: -11.8499851, -9.2783871, -11.8321257, -9.3152046, -1.7564399, 1.7405159

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6019677, upper bound: 0.6098014
time: 7.65 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019676, upper bound: 0.6073027
time: 6.24 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.1252279, -10.4172678, -13.1749344, -10.4130869, -1.8065784, 1.8226171
1: -7.1643620, -4.1769524, -7.1811538, -4.1630096, -2.3612676, 2.3504698
2: 9.3269081, 11.2836542, 9.3147564, 11.3186913, -1.5739193, 1.5762956
3: -4.8956337, -2.7368512, -4.9083776, -2.7331634, -1.9421091, 1.9482839
4: -9.4387159, -6.7140183, -9.4542303, -6.7051563, -1.9791012, 1.9780440
5: -13.8032322, -11.1587372, -13.8620758, -11.1566887, -1.6296351, 1.6511505
6: -16.3591061, -12.7423735, -16.3818703, -12.7188129, -2.3440619, 2.3238635
7: -4.0620880, -1.3630304, -4.0922680, -1.3343673, -2.4898529, 2.4888561
8: -6.0698185, -3.6185808, -6.1327572, -3.6177726, -2.0494876, 2.0704737
9: -11.8503208, -9.2783871, -11.8667297, -9.2650146, -1.7610724, 1.7639437

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6019677, upper bound: 0.6098882
time: 6.22 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019676, upper bound: 0.6073876
time: 7.95 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -13.1263924, -10.4119873, -13.1521168, -10.4660330, -1.8041177, 1.8061686
1: -7.1653781, -4.1722422, -7.1521473, -4.1804566, -2.3410954, 2.3474483
2: 9.3232517, 11.2848692, 9.3525238, 11.3094940, -1.5712166, 1.5685596
3: -4.8965383, -2.7342093, -4.8874712, -2.7352757, -1.9398701, 1.9366202
4: -9.4410782, -6.7113867, -9.4574366, -6.7167521, -1.9667110, 1.9863653
5: -13.8050404, -11.1584806, -13.8552999, -11.1738110, -1.6011691, 1.6324477
6: -16.3596897, -12.7377205, -16.3680344, -12.7265644, -2.3265238, 2.3196833
7: -4.0663877, -1.3625522, -4.0893741, -1.3415108, -2.4913473, 2.4885690
8: -6.0710344, -3.6128311, -6.1061368, -3.6197844, -2.0327935, 2.0568905
9: -11.8567619, -9.2782364, -11.8473902, -9.3099794, -1.7653551, 1.7470400

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4586

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6111791
time: 6.56 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6086791
time: 6.60 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.1264286, -10.4119844, -13.1842270, -10.4024887, -1.8097150, 1.8335544
1: -7.1656537, -4.1722298, -7.1900778, -4.1537890, -2.3659759, 2.3601351
2: 9.3224754, 11.2848854, 9.3052368, 11.3269148, -1.5843048, 1.5802650
3: -4.8966589, -2.7341857, -4.9147048, -2.7281127, -1.9480135, 1.9535279
4: -9.4410849, -6.7102261, -9.4625893, -6.6975613, -1.9865232, 1.9935732
5: -13.8050423, -11.1570950, -13.8692093, -11.1530228, -1.6327424, 1.6575465
6: -16.3600769, -12.7376862, -16.3929119, -12.7067070, -2.3496647, 2.3337240
7: -4.0664120, -1.3625369, -4.1019597, -1.3324368, -2.4991398, 2.5008917
8: -6.0710640, -3.6127234, -6.1426477, -3.6058135, -2.0533338, 2.0838795
9: -11.8570995, -9.2782345, -11.8822594, -9.2597513, -1.7707124, 1.7718832

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4586

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6112673
time: 4.77 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6087636
time: 6.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 26.10 seconds
IS_A2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6019677, upper bound: 0.6098014
IS_A2_B2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6019676, upper bound: 0.6073027
IS_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6019677, upper bound: 0.6098882
IS_A2_B2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6019676, upper bound: 0.6073876
IS_A2_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6111791
IS_A2_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6086791
IS_A2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6112673
IS_A2_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 26.10
Output dim: 2, lower bound: -0.6021115, upper bound: 0.6087636

## BFS IS instance: IS_A2_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1222906, -10.4190502, -13.1431446, -10.4762869, -1.7987974, 1.7932582
1: -7.1582355, -4.1786270, -7.1440210, -4.1896644, -2.3305306, 2.3341713
2: 9.3402271, 11.2816343, 9.3618107, 11.3021851, -1.5484972, 1.5620201
3: -4.8934050, -2.7392387, -4.8813162, -2.7403073, -1.9319313, 1.9225802
4: -9.4327431, -6.7184620, -9.4495335, -6.7242875, -1.9515877, 1.9661500
5: -13.8018408, -11.1816864, -13.8491621, -11.1774254, -1.6078129, 1.6047313
6: -16.3569126, -12.7475519, -16.3588028, -12.7386227, -2.3152742, 2.3039141
7: -4.0371222, -1.3653452, -4.0797834, -1.3433249, -2.4539733, 2.4686570
8: -6.0653329, -3.6409750, -6.0963173, -3.6311989, -2.0245714, 2.0207677
9: -11.8372297, -9.2797136, -11.8321257, -9.3152046, -1.7430499, 1.7375777

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5985110, upper bound: 0.6097641
time: 9.19 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5985110, upper bound: 0.6064436
time: 7.53 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1223259, -10.4190512, -13.1749344, -10.4130869, -1.8027532, 1.8204818
1: -7.1585097, -4.1786151, -7.1811538, -4.1630096, -2.3553753, 2.3466251
2: 9.3394489, 11.2816505, 9.3147564, 11.3186913, -1.5608325, 1.5759883
3: -4.8935261, -2.7392151, -4.9083776, -2.7331634, -1.9385839, 1.9420295
4: -9.4327507, -6.7173023, -9.4542303, -6.7051563, -1.9713140, 1.9724734
5: -13.8018446, -11.1802998, -13.8620758, -11.1566887, -1.6272242, 1.6294039
6: -16.3572998, -12.7475157, -16.3818703, -12.7188129, -2.3383603, 2.3169401
7: -4.0371466, -1.3653297, -4.0922680, -1.3343673, -2.4616179, 2.4809251
8: -6.0653620, -3.6408658, -6.1327572, -3.6177726, -2.0429254, 2.0477395
9: -11.8375683, -9.2797127, -11.8667297, -9.2650146, -1.7476790, 1.7651167

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986899, upper bound: 0.6098487
time: 7.08 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5986898, upper bound: 0.6064282
time: 9.44 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1234932, -10.4137707, -13.1521168, -10.4660330, -1.8002925, 1.8040029
1: -7.1595249, -4.1739039, -7.1521473, -4.1804566, -2.3351970, 2.3435993
2: 9.3357964, 11.2828655, 9.3525238, 11.3094940, -1.5581284, 1.5648351
3: -4.8944273, -2.7365761, -4.8874712, -2.7352757, -1.9357007, 1.9299979
4: -9.4350996, -6.7146692, -9.4574366, -6.7167521, -1.9588909, 1.9807711
5: -13.8036528, -11.1800404, -13.8552999, -11.1738110, -1.6109087, 1.6106944
6: -16.3578835, -12.7428665, -16.3680344, -12.7265644, -2.3208261, 2.3127596
7: -4.0414152, -1.3648553, -4.0893741, -1.3415108, -2.4630466, 2.4806476
8: -6.0665779, -3.6351156, -6.1061368, -3.6197844, -2.0280981, 2.0341587
9: -11.8440094, -9.2795620, -11.8473902, -9.3099794, -1.7519646, 1.7441015

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4586

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986639, upper bound: 0.6111408
time: 13.36 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5986638, upper bound: 0.6077341
time: 5.21 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1235237, -10.4137669, -13.1842270, -10.4024887, -1.8058903, 1.8314190
1: -7.1597996, -4.1738944, -7.1900778, -4.1537890, -2.3600802, 2.3562808
2: 9.3350220, 11.2828817, 9.3052368, 11.3269148, -1.5712166, 1.5790188
3: -4.8945484, -2.7365518, -4.9147048, -2.7281127, -1.9438479, 1.9486351
4: -9.4351053, -6.7135105, -9.4625893, -6.6975613, -1.9786992, 1.9879780
5: -13.8036499, -11.1786575, -13.8692093, -11.1530228, -1.6303287, 1.6357937
6: -16.3582726, -12.7428322, -16.3929119, -12.7067070, -2.3439641, 2.3267994
7: -4.0414400, -1.3648405, -4.1019597, -1.3324368, -2.4708390, 2.4929705
8: -6.0666075, -3.6350069, -6.1426477, -3.6058135, -2.0467710, 2.0611484
9: -11.8443451, -9.2795620, -11.8822594, -9.2597513, -1.7573218, 1.7719808

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4586

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5988428, upper bound: 0.6112292
time: 13.08 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5988427, upper bound: 0.6078195
time: 6.33 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 33.73 seconds
IS_A2_B2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5985110, upper bound: 0.6097641
IS_A2_B2_B2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5985110, upper bound: 0.6064436
IS_A2_B2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5986899, upper bound: 0.6098487
IS_A2_B2_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5986898, upper bound: 0.6064282
IS_A2_B2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5986639, upper bound: 0.6111408
IS_A2_B2_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5986638, upper bound: 0.6077341
IS_A2_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5988428, upper bound: 0.6112292
IS_A2_B2_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 33.73
Output dim: 2, lower bound: -0.5988427, upper bound: 0.6078195

## BFS IS instance: IS_A2_B2_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -13.1211510, -10.4321337, -13.1431360, -10.4762974, -1.7974041, 1.7786593
1: -7.1572084, -4.1844630, -7.1440020, -4.1898999, -2.3219626, 2.3232841
2: 9.3480206, 11.2803116, 9.3618145, 11.3020840, -1.5397561, 1.5584817
3: -4.8925934, -2.7416923, -4.8812723, -2.7403083, -1.9289145, 1.9182644
4: -9.4305220, -6.7256689, -9.4495335, -6.7243042, -1.9457984, 1.9583471
5: -13.8013287, -11.1915951, -13.8488064, -11.1774311, -1.6061716, 1.5942757
6: -16.3562775, -12.7511549, -16.3587780, -12.7386551, -2.3127398, 2.2972760
7: -4.0330563, -1.3719754, -4.0796857, -1.3433285, -2.4473367, 2.4617910
8: -6.0637670, -3.6501784, -6.0959749, -3.6311998, -2.0218711, 2.0112870
9: -11.8344364, -9.2813654, -11.8320370, -9.3152008, -1.7395146, 1.7358904

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 6218
type: B, layer: 1, pos: 6218
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5980217, upper bound: 0.6097371
time: 6.76 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5984851, upper bound: 0.6097354
time: 10.04 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -13.1211843, -10.4321337, -13.1749268, -10.4130964, -1.8034623, 1.8058796
1: -7.1574845, -4.1844535, -7.1811342, -4.1632438, -2.3419504, 2.3357422
2: 9.3472443, 11.2803259, 9.3147602, 11.3185873, -1.5520911, 1.5724156
3: -4.8927145, -2.7416685, -4.9083343, -2.7331657, -1.9292145, 1.9377255
4: -9.4305258, -6.7245116, -9.4542284, -6.7051716, -1.9654508, 1.9646661
5: -13.8013287, -11.1902065, -13.8617220, -11.1566906, -1.6255999, 1.6189353
6: -16.3566666, -12.7511225, -16.3818436, -12.7188492, -2.3357506, 2.3103125
7: -4.0330830, -1.3719609, -4.0921698, -1.3343730, -2.4549828, 2.4740596
8: -6.0637960, -3.6500683, -6.1324148, -3.6177754, -2.0402412, 2.0382566
9: -11.8347750, -9.2813663, -11.8666410, -9.2650146, -1.7437234, 1.7634256

Time for backsubstitution: 14.22 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.56638765335083
rel_dist={2: [-0.6118841057361983, 0.6118843662850555]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2431.21 seconds
