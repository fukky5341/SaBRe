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
execution time: IAR + LP analysis = 15.05 + 32.62 = 47.67 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.33 seconds, max iter: 100)

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
Binary search time: 213.27 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3339.06 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328291, upper bound: 1.0193407
time: 6.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328289, upper bound: 1.0328299
time: 6.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.83
Output dim: 2, lower bound: -1.0328291, upper bound: 1.0193407
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.83
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

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193392
time: 9.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193390
time: 10.52 seconds

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

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0328285
time: 10.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0328308
time: 11.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 34.25 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 34.25
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193392
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.25
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0193390
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.25
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0328285
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.25
Output dim: 2, lower bound: -1.0193388, upper bound: 1.0328308

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

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192373
time: 7.83 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
time: 8.68 seconds

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

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192352
time: 7.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
time: 7.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.1264086, -10.4119644, -13.0942993, -10.4755249, -2.2346478, 2.2202609
1: -7.1655025, -4.1722221, -7.1278400, -4.1993065, -2.7422686, 2.7299867
2: 9.3228989, 11.2848797, 9.3696823, 11.2674522, -1.8734753, 1.8655391
3: -4.8965921, -2.7341909, -4.8695817, -2.7414377, -2.1551545, 2.1353908
4: -9.4410906, -6.7108707, -9.4359188, -6.7293978, -2.3667374, 2.3818431
5: -13.8050432, -11.1578493, -13.7910433, -11.1777372, -2.0140533, 2.0193682
6: -16.3598595, -12.7376928, -16.3353348, -12.7582769, -2.7933426, 2.7897968
7: -4.0664282, -1.3625374, -4.0544095, -1.3716025, -2.6948256, 2.6918721
8: -6.0710535, -3.6127505, -6.0347061, -3.6268172, -2.3310919, 2.3083701
9: -11.8569212, -9.2782335, -11.8219433, -9.3285122, -2.1737175, 2.1615009

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117998, upper bound: 1.0327166
time: 6.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193250, upper bound: 1.0328146
time: 8.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.1264305, -10.4119644, -13.1264305, -10.4119644, -2.2512021, 2.2512021
1: -7.1656604, -4.1722155, -7.1656604, -4.1722155, -2.7597809, 2.7597809
2: 9.3224525, 11.2848892, 9.3224525, 11.2848892, -1.8809371, 1.8809369
3: -4.8966618, -2.7341783, -4.8966618, -2.7341783, -2.1624835, 2.1624835
4: -9.4410954, -6.7102103, -9.4410954, -6.7102103, -2.3893867, 2.3893869
5: -13.8050451, -11.1570587, -13.8050451, -11.1570587, -2.0638740, 2.0638738
6: -16.3600807, -12.7376738, -16.3600807, -12.7376738, -2.8174701, 2.8174701
7: -4.0664425, -1.3625278, -4.0664425, -1.3625278, -2.7039146, 2.7039146
8: -6.0710688, -3.6126885, -6.0710688, -3.6126885, -2.3584661, 2.3584657
9: -11.8571129, -9.2782326, -11.8571129, -9.2782326, -2.1841836, 2.1841836

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117998, upper bound: 1.0327176
time: 8.18 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193250, upper bound: 1.0328144
time: 6.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192373
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0118041, upper bound: 1.0192352
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0193293, upper bound: 1.0193254
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0117998, upper bound: 1.0327166
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0193250, upper bound: 1.0328146
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0117998, upper bound: 1.0327176
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 2, lower bound: -1.0193250, upper bound: 1.0328144

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

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0123389
time: 5.08 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0192270
time: 5.63 seconds

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

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0124307
time: 6.16 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0193152
time: 8.79 seconds

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

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0252637, upper bound: 1.0123367
time: 6.26 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0252637, upper bound: 1.0192228
time: 5.05 seconds

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

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0124268
time: 5.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0193109
time: 6.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1235094, -10.4137506, -13.0936127, -10.4759321, -2.2303946, 2.2173073
1: -7.1596451, -4.1738858, -7.1264777, -4.1996861, -2.7360792, 2.7274323
2: 9.3354530, 11.2828770, 9.3726139, 11.2669907, -1.8594112, 1.8590062
3: -4.8944807, -2.7365592, -4.8690891, -2.7419920, -2.1524887, 2.1325300
4: -9.4351072, -6.7141566, -9.4345217, -6.7301540, -2.3582563, 2.3769846
5: -13.8036499, -11.1794338, -13.7907248, -11.1827641, -2.0077095, 1.9973314
6: -16.3580513, -12.7428446, -16.3349190, -12.7594805, -2.7865357, 2.7816458
7: -4.0414348, -1.3648405, -4.0485830, -1.3721285, -2.6693063, 2.6837425
8: -6.0665941, -3.6350532, -6.0336771, -3.6320171, -2.3215399, 2.2846799
9: -11.8441620, -9.2795620, -11.8189669, -9.3288155, -2.1600912, 2.1563499

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117857, upper bound: 1.0258269
time: 5.24 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117857, upper bound: 1.0327045
time: 5.14 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1298571, -10.4080219, -13.0943031, -10.4755287, -2.2380550, 2.2215652
1: -7.1661758, -4.1653981, -7.1278396, -4.1993051, -2.7427087, 2.7366714
2: 9.3204021, 11.3007107, 9.3696890, 11.2674513, -1.8719316, 1.8800440
3: -4.8981237, -2.7331772, -4.8695817, -2.7414374, -2.1566863, 2.1364045
4: -9.4423256, -6.7032957, -9.4359159, -6.7293987, -2.3671188, 2.3894930
5: -13.8303528, -11.1573315, -13.7910433, -11.1777439, -2.0395732, 2.0129390
6: -16.3657627, -12.7350721, -16.3353348, -12.7582769, -2.7965064, 2.7929919
7: -4.0701303, -1.3357048, -4.0543995, -1.3716013, -2.6985290, 2.7186947
8: -6.0994263, -3.6104364, -6.0347042, -3.6268263, -2.3466606, 2.3045702
9: -11.8592920, -9.2678957, -11.8219395, -9.3285122, -2.1737618, 2.1638739

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193109, upper bound: 1.0259265
time: 8.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193109, upper bound: 1.0328001
time: 7.22 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1235275, -10.4137487, -13.1257496, -10.4123716, -2.2469437, 2.2484241
1: -7.1598024, -4.1738791, -7.1642942, -4.1725979, -2.7536001, 2.7572422
2: 9.3350086, 11.2828846, 9.3253765, 11.2844267, -1.8670483, 1.8744020
3: -4.8945503, -2.7365460, -4.8961620, -2.7347319, -2.1598184, 2.1596160
4: -9.4351110, -6.7134953, -9.4396963, -6.7109618, -2.3809023, 2.3845236
5: -13.8036537, -11.1786404, -13.8047218, -11.1620865, -2.0548906, 2.0418108
6: -16.3582745, -12.7428226, -16.3596592, -12.7388792, -2.8106618, 2.8093195
7: -4.0414495, -1.3648298, -4.0606165, -1.3630548, -2.6783948, 2.6957867
8: -6.0666089, -3.6349907, -6.0700397, -3.6178827, -2.3489165, 2.3347769
9: -11.8443556, -9.2795610, -11.8541355, -9.2785397, -2.1705570, 2.1800621

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0122795, upper bound: 1.0123345
time: 10.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0122795, upper bound: 1.0327032
time: 10.06 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1298771, -10.4080219, -13.1264305, -10.4119644, -2.2546678, 2.2538595
1: -7.1663308, -4.1653929, -7.1656585, -4.1722164, -2.7602177, 2.7664728
2: 9.3199577, 11.3007202, 9.3224621, 11.2848864, -1.8806434, 1.8944254
3: -4.8981905, -2.7331629, -4.8966608, -2.7341771, -2.1640134, 2.1634979
4: -9.4423294, -6.7026329, -9.4410915, -6.7102094, -2.3897724, 2.3970277
5: -13.8303518, -11.1565361, -13.8050442, -11.1570702, -2.0658662, 2.0574744
6: -16.3659859, -12.7350512, -16.3600826, -12.7376785, -2.8206334, 2.8206952
7: -4.0701437, -1.3356965, -4.0664315, -1.3625276, -2.7076161, 2.7307351
8: -6.0994430, -3.6103745, -6.0710683, -3.6126986, -2.3668904, 2.3546658
9: -11.8594837, -9.2678967, -11.8571091, -9.2782335, -2.1842289, 2.1937900

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0198062, upper bound: 1.0259272
time: 5.30 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0198062, upper bound: 1.0328013
time: 6.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0123389
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0117900, upper bound: 1.0192270
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0124307
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0193152, upper bound: 1.0193152
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0252637, upper bound: 1.0123367
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0252637, upper bound: 1.0192228
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0124268
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0328005, upper bound: 1.0193109
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0117857, upper bound: 1.0258269
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0117857, upper bound: 1.0327045
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0193109, upper bound: 1.0259265
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0193109, upper bound: 1.0328001
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0122795, upper bound: 1.0123345
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0122795, upper bound: 1.0327032
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0198062, upper bound: 1.0259272
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 2, lower bound: -1.0198062, upper bound: 1.0328013

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

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0123390
time: 6.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0123386
time: 8.08 seconds

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

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0192256
time: 9.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049052, upper bound: 1.0192254
time: 6.10 seconds

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

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0124328
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0124308
time: 6.92 seconds

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

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0193175
time: 5.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0193160
time: 6.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.0913610, -10.4773083, -13.1245928, -10.4254808, -2.2013638, 2.2311220
1: -7.1219940, -4.2009621, -7.1631169, -4.1784763, -2.7128758, 2.7343211
2: 9.3822632, 11.2654476, 9.3336296, 11.2831163, -1.8487968, 1.8567128
3: -4.8674974, -2.7438030, -4.8952894, -2.7372203, -2.1302772, 2.1514864
4: -9.4299364, -6.7326961, -9.4374828, -6.7188473, -2.3655577, 2.3549333
5: -13.7896509, -11.1993170, -13.8042145, -11.1728077, -2.0028744, 1.9915514
6: -16.3335457, -12.7634296, -16.3588047, -12.7425289, -2.7762489, 2.7846713
7: -4.0294218, -1.3739083, -4.0565176, -1.3697047, -2.6597171, 2.6826093
8: -6.0302114, -3.6491370, -6.0684757, -3.6271596, -2.2896147, 2.3056178
9: -11.8091946, -9.3298378, -11.8511486, -9.2801924, -2.1462429, 2.1661630

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0123348
time: 14.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0123352
time: 5.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.0913591, -10.4773159, -13.1800842, -10.4068317, -2.2235892, 2.2646365
1: -7.1219931, -4.2009649, -7.1878791, -4.1609902, -2.7526269, 2.7451890
2: 9.3822680, 11.2654467, 9.3110476, 11.3106117, -1.8678522, 1.8837171
3: -4.8674970, -2.7438049, -4.9126205, -2.7296734, -2.1378236, 2.1688156
4: -9.4299345, -6.7327027, -9.4599771, -6.7065487, -2.3819213, 2.3788319
5: -13.7896500, -11.1993246, -13.8435841, -11.1593542, -2.0172577, 2.0251021
6: -16.3335438, -12.7634315, -16.3863811, -12.7105045, -2.8271322, 2.8009241
7: -4.0294199, -1.3739123, -4.0924511, -1.3597929, -2.6696270, 2.7185388
8: -6.0302114, -3.6491418, -6.1132479, -3.6133366, -2.3055248, 2.3326075
9: -11.8091946, -9.3298397, -11.8767357, -9.2703848, -2.1560988, 2.1957533

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192233
time: 5.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192215
time: 7.33 seconds

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

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259242, upper bound: 1.0124268
time: 6.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0124264
time: 7.03 seconds

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

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0193109
time: 12.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0193116
time: 8.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1235094, -10.4137506, -13.0925255, -10.4890633, -2.2158380, 2.2166119
1: -7.1596451, -4.1738858, -7.1254659, -4.2055492, -2.7251592, 2.7220092
2: 9.3354530, 11.2828770, 9.3804674, 11.2657003, -1.8565273, 1.8503351
3: -4.8944807, -2.7365592, -4.8682957, -2.7444711, -2.1500096, 2.1317365
4: -9.4351072, -6.7141566, -9.4323273, -6.7373934, -2.3504562, 2.3700318
5: -13.8036499, -11.1794338, -13.7902412, -11.1927404, -1.9975195, 1.9968922
6: -16.3580513, -12.7428446, -16.3343010, -12.7631636, -2.7799091, 2.7811334
7: -4.0414348, -1.3648405, -4.0445247, -1.3787723, -2.6626625, 2.6796842
8: -6.0665941, -3.6350532, -6.0321326, -3.6412392, -2.3123322, 2.2828693
9: -11.8441620, -9.2795620, -11.8162031, -9.3304996, -2.1584733, 2.1529529

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049008, upper bound: 1.0258247
time: 7.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049008, upper bound: 1.0258268
time: 6.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1235085, -10.4137554, -13.1479492, -10.4703770, -2.2416279, 2.2322679
1: -7.1596432, -4.1738901, -7.1501160, -4.1876540, -2.7536545, 2.7442627
2: 9.3354568, 11.2828751, 9.3578920, 11.2931919, -1.8619370, 1.8816361
3: -4.8944812, -2.7365613, -4.8854609, -2.7368197, -2.1576614, 2.1488996
4: -9.4351072, -6.7141633, -9.4548302, -6.7250795, -2.3667464, 2.3939657
5: -13.8036537, -11.1794386, -13.8296738, -11.1793470, -2.0118227, 2.0172596
6: -16.3580513, -12.7428474, -16.3617477, -12.7303581, -2.8250539, 2.8095329
7: -4.0414314, -1.3648436, -4.0798874, -1.3688598, -2.6725717, 2.7150438
8: -6.0665927, -3.6350579, -6.0767307, -3.6272840, -2.3282943, 2.3275721
9: -11.8441620, -9.2795630, -11.8420448, -9.3206024, -2.1691635, 2.1767478

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049008, upper bound: 1.0327029
time: 8.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0049009, upper bound: 1.0327049
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1298571, -10.4080219, -13.0932150, -10.4886551, -2.2235003, 2.2208683
1: -7.1661758, -4.1653981, -7.1268258, -4.2051682, -2.7317891, 2.7312460
2: 9.3204021, 11.3007107, 9.3775463, 11.2661619, -1.8690445, 1.8713732
3: -4.8981237, -2.7331772, -4.8687849, -2.7439201, -2.1542037, 2.1356077
4: -9.4423256, -6.7032957, -9.4337177, -6.7366381, -2.3593230, 2.3825352
5: -13.8303528, -11.1573315, -13.7905636, -11.1877251, -2.0293832, 2.0124993
6: -16.3657627, -12.7350721, -16.3347187, -12.7619610, -2.7898793, 2.7924809
7: -4.0701303, -1.3357048, -4.0503359, -1.3782458, -2.6918845, 2.7146311
8: -6.0994263, -3.6104364, -6.0331645, -3.6360478, -2.3374534, 2.3027616
9: -11.8592920, -9.2678957, -11.8191776, -9.3301983, -2.1721444, 2.1604769

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0259265
time: 5.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0259241
time: 8.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1298552, -10.4080305, -13.1486273, -10.4699688, -2.2492847, 2.2365108
1: -7.1661739, -4.1654019, -7.1514769, -4.1872692, -2.7602758, 2.7535043
2: 9.3204079, 11.3007107, 9.3549767, 11.2936497, -1.8744578, 1.8985868
3: -4.8981209, -2.7331784, -4.8859525, -2.7362623, -2.1618586, 2.1527741
4: -9.4423246, -6.7033005, -9.4562149, -6.7243204, -2.3755932, 2.4047372
5: -13.8303528, -11.1573372, -13.8299971, -11.1743279, -2.0436869, 2.0329251
6: -16.3657570, -12.7350731, -16.3621635, -12.7291574, -2.8325324, 2.8197184
7: -4.0701284, -1.3357096, -4.0856819, -1.3683290, -2.7017994, 2.7499723
8: -6.0994253, -3.6104412, -6.0777607, -3.6220970, -2.3535066, 2.3475442
9: -11.8592911, -9.2678986, -11.8449965, -9.3202982, -2.1828341, 2.1842661

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0328027
time: 6.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0328028
time: 6.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1235275, -10.4137487, -13.1246128, -10.4254780, -2.2323663, 2.2477016
1: -7.1598024, -4.1738791, -7.1632733, -4.1784716, -2.7427306, 2.7518272
2: 9.3350086, 11.2828846, 9.3331871, 11.2831268, -1.8641891, 1.8657150
3: -4.8945503, -2.7365460, -4.8953571, -2.7372065, -2.1573439, 2.1588111
4: -9.4351110, -6.7134953, -9.4374876, -6.7181864, -2.3730869, 2.3775780
5: -13.8036537, -11.1786404, -13.8042126, -11.1720181, -2.0446825, 2.0413797
6: -16.3582745, -12.7428226, -16.3590260, -12.7425098, -2.8040180, 2.8087890
7: -4.0414495, -1.3648298, -4.0565319, -1.3696947, -2.6717548, 2.6917021
8: -6.0666089, -3.6349907, -6.0684929, -3.6270971, -2.3397198, 2.3330064
9: -11.8443556, -9.2795610, -11.8513422, -9.2801895, -2.1689305, 2.1766250

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0053976, upper bound: 1.0258246
time: 6.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0053976, upper bound: 1.0258252
time: 5.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.34 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0123390
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0123386
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049051, upper bound: 1.0192256
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049052, upper bound: 1.0192254
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0124328
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0124308
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0193175
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124305, upper bound: 1.0193160
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0123348
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0123352
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192233
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0183997, upper bound: 1.0192215
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0259242, upper bound: 1.0124268
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0124264
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0193109
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0259243, upper bound: 1.0193116
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049008, upper bound: 1.0258247
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049008, upper bound: 1.0258268
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049008, upper bound: 1.0327029
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0049009, upper bound: 1.0327049
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0259265
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0259241
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0328027
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0124262, upper bound: 1.0328028
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0053976, upper bound: 1.0258246
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.34
Output dim: 2, lower bound: -1.0053976, upper bound: 1.0258252
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.34
Output dim: 2, lower bound: -1.0122795, upper bound: 1.0327032
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.34
Output dim: 2, lower bound: -1.0198062, upper bound: 1.0259272
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.34
Output dim: 2, lower bound: -1.0198062, upper bound: 1.0328013
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8640661239624023
rel_dist={2: [-1.032862430954955, 1.0328621621306109]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7217719, upper bound: 0.7140570
time: 8.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218423, upper bound: 0.7218413
time: 9.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.73
Output dim: 2, lower bound: -0.7217719, upper bound: 0.7140570
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.73
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

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140574, upper bound: 0.7140595
time: 6.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140574, upper bound: 0.7140578
time: 7.57 seconds

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

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7213037, upper bound: 0.7168645
time: 7.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218339, upper bound: 0.7218322
time: 9.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.39 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 2, lower bound: -0.7140574, upper bound: 0.7140595
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 2, lower bound: -0.7140574, upper bound: 0.7140578
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 2, lower bound: -0.7213037, upper bound: 0.7168645
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 2, lower bound: -0.7218339, upper bound: 0.7218322

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.0942993, -10.4755249, -1.8808722, 1.8808722
1: -7.1278400, -4.1993065, -7.1278400, -4.1993065, -2.4078102, 2.4078097
2: 9.3696823, 11.2674522, 9.3696823, 11.2674522, -1.6227055, 1.6227055
3: -4.8695817, -2.7414377, -4.8695817, -2.7414377, -1.9904337, 1.9904337
4: -9.4359188, -6.7293978, -9.4359188, -6.7293978, -2.0560503, 2.0560503
5: -13.7910433, -11.1777372, -13.7910433, -11.1777372, -1.7020273, 1.7020273
6: -16.3353348, -12.7582769, -16.3353348, -12.7582769, -2.4018059, 2.4018054
7: -4.0544095, -1.3716025, -4.0544095, -1.3716025, -2.5354190, 2.5354190
8: -6.0347061, -3.6268172, -6.0347061, -3.6268172, -2.0824118, 2.0824118
9: -11.8219433, -9.3285122, -11.8219433, -9.3285122, -1.8226118, 1.8226113

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090763, upper bound: 0.7135133
time: 10.54 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140510, upper bound: 0.7140491
time: 6.17 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.0942993, -10.4755249, -13.1264038, -10.4119644, -1.8998890, 1.9169543
1: -7.1278400, -4.1993065, -7.1654177, -4.1722250, -2.4316726, 2.4413114
2: 9.3696823, 11.2674522, 9.3231411, 11.2848749, -1.6422715, 1.6469789
3: -4.8695817, -2.7414377, -4.8965569, -2.7341986, -1.9986243, 2.0191054
4: -9.4359188, -6.7293978, -9.4410877, -6.7112317, -2.0766454, 2.0618291
5: -13.7910433, -11.1777372, -13.8050442, -11.1582813, -1.7208862, 1.7159281
6: -16.3353348, -12.7582769, -16.3597374, -12.7377071, -2.4236717, 2.4270077
7: -4.0544095, -1.3716025, -4.0664206, -1.3625400, -2.5458469, 2.5486269
8: -6.0347061, -3.6268172, -6.0710440, -3.6127853, -2.0969882, 2.1197138
9: -11.8219433, -9.3285122, -11.8568172, -9.2782345, -1.8434374, 1.8591380

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090763, upper bound: 0.7135155
time: 7.07 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140510, upper bound: 0.7140496
time: 7.06 seconds

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

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7167348, upper bound: 0.7168374
time: 6.71 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7212988, upper bound: 0.7168608
time: 7.94 seconds

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
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7172645, upper bound: 0.7218059
time: 9.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218290, upper bound: 0.7218284
time: 7.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 31.30 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7090763, upper bound: 0.7135133
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7140510, upper bound: 0.7140491
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7090763, upper bound: 0.7135155
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7140510, upper bound: 0.7140496
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7167348, upper bound: 0.7168374
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7212988, upper bound: 0.7168608
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7172645, upper bound: 0.7218059
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 2, lower bound: -0.7218290, upper bound: 0.7218284

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.0913610, -10.4773083, -13.0928650, -10.4763803, -1.8761001, 1.8771088
1: -7.1219940, -4.2009621, -7.1250038, -4.2000990, -2.4013538, 2.4037943
2: 9.3822632, 11.2654476, 9.3757820, 11.2664871, -1.6079412, 1.6128812
3: -4.8674974, -2.7438030, -4.8685598, -2.7425900, -1.9847512, 1.9843030
4: -9.4299364, -6.7326961, -9.4330139, -6.7309794, -2.0468068, 2.0492320
5: -13.7896509, -11.1993170, -13.7903738, -11.1882000, -1.6902051, 1.6796613
6: -16.3335457, -12.7634296, -16.3344650, -12.7607803, -2.3932471, 2.3923495
7: -4.0294218, -1.3739083, -4.0422869, -1.3727078, -2.5035715, 2.5144615
8: -6.0302114, -3.6491370, -6.0325508, -3.6376381, -2.0671272, 2.0576358
9: -11.8091946, -9.3298378, -11.8157520, -9.3291473, -1.8087468, 1.8151243

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090476, upper bound: 0.7089447
time: 8.49 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7090720, upper bound: 0.7135107
time: 7.96 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715805, -13.0943031, -10.4755249, -1.8841197, 1.8835294
1: -7.1285033, -4.1924777, -7.1278381, -4.1993060, -2.4080200, 2.4144983
2: 9.3671894, 11.2832909, 9.3696928, 11.2674522, -1.6189070, 1.6341453
3: -4.8711004, -2.7404246, -4.8695812, -2.7414370, -1.9934239, 1.9893780
4: -9.4371519, -6.7218237, -9.4359159, -6.7293987, -2.0561976, 2.0637083
5: -13.8163528, -11.1772184, -13.7910461, -11.1777534, -1.7216158, 1.6902709
6: -16.3412170, -12.7556314, -16.3353329, -12.7582788, -2.4049253, 2.4035220
7: -4.0580997, -1.3447695, -4.0543909, -1.3716025, -2.5277758, 2.5572166
8: -6.0630856, -3.6244631, -6.0347033, -3.6268330, -2.1092725, 2.0738845
9: -11.8243227, -9.3181829, -11.8219376, -9.3285141, -1.8211360, 1.8321934

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140230, upper bound: 0.7094841
time: 11.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140466, upper bound: 0.7140462
time: 8.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.0913610, -10.4773083, -13.1249905, -10.4128208, -1.8951056, 1.9132128
1: -7.1219940, -4.2009621, -7.1625757, -4.1730232, -2.4252100, 2.4366357
2: 9.3822632, 11.2654476, 9.3292265, 11.2839088, -1.6275077, 1.6353254
3: -4.8674974, -2.7438030, -4.8955221, -2.7353508, -1.9929399, 2.0129638
4: -9.4299364, -6.7326961, -9.4381828, -6.7128096, -2.0674019, 2.0550132
5: -13.7896509, -11.1993170, -13.8043737, -11.1687450, -1.7090626, 1.6935647
6: -16.3335457, -12.7634296, -16.3588638, -12.7402105, -2.4151092, 2.4175584
7: -4.0294218, -1.3739083, -4.0542965, -1.3636456, -2.5140018, 2.5276632
8: -6.0302114, -3.6491370, -6.0688901, -3.6235971, -2.0817099, 2.0949426
9: -11.8091946, -9.3298378, -11.8506165, -9.2788696, -1.8295703, 1.8516448

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7167642, upper bound: 0.7089425
time: 14.58 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7167895, upper bound: 0.7135080
time: 9.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.0977631, -10.4715805, -13.1264009, -10.4119635, -1.9031410, 1.9196103
1: -7.1285033, -4.1924777, -7.1654148, -4.1722269, -2.4318824, 2.4433560
2: 9.3671894, 11.2832909, 9.3231506, 11.2848740, -1.6384735, 1.6478958
3: -4.8711004, -2.7404246, -4.8965549, -2.7341995, -2.0016141, 2.0180497
4: -9.4371519, -6.7218237, -9.4410839, -6.7112331, -2.0767927, 2.0694880
5: -13.8163528, -11.1772184, -13.8050451, -11.1582966, -1.7273302, 1.7041709
6: -16.3412170, -12.7556314, -16.3597374, -12.7377090, -2.4267912, 2.4280272
7: -4.0580997, -1.3447695, -4.0664034, -1.3625422, -2.5382042, 2.5693040
8: -6.0630856, -3.6244631, -6.0710421, -3.6127996, -2.1216125, 2.1111875
9: -11.8243227, -9.3181829, -11.8568096, -9.2782345, -1.8419838, 1.8687201

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7217333, upper bound: 0.7094791
time: 8.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7217578, upper bound: 0.7140442
time: 8.23 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1238670, -10.4259233, -13.1141987, -10.4803190, -1.9115701, 1.9056230
1: -7.1617045, -4.1788907, -7.1231785, -4.1881351, -2.4470558, 2.4312172
2: 9.3366070, 11.2826157, 9.3823214, 11.2790012, -1.6415687, 1.6198390
3: -4.8947821, -2.7378101, -4.8696766, -2.7394078, -2.0181656, 1.9987407
4: -9.4359779, -6.7193975, -9.4321976, -6.7299905, -2.0597830, 2.0634866
5: -13.8038635, -11.1779137, -13.7963238, -11.1990061, -1.7243693, 1.7146947
6: -16.3584404, -12.7438183, -16.3356094, -12.7611895, -2.4209061, 2.4176447
7: -4.0502276, -1.3702762, -4.0302968, -1.3736777, -2.5251679, 2.5085845
8: -6.0673571, -3.6327515, -6.0326695, -3.6441708, -2.1072769, 2.0785699
9: -11.8480053, -9.2805252, -11.8294058, -9.3296642, -1.8422227, 1.8485999

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7167646
time: 5.98 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7168382
time: 4.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1793556, -10.4072800, -13.1144810, -10.4769850, -1.9389367, 1.9212320
1: -7.1864686, -4.1614060, -7.1234317, -4.1866040, -2.4567199, 2.4653213
2: 9.3140163, 11.3101158, 9.3803329, 11.2793264, -1.6651120, 1.6380284
3: -4.9121103, -2.7302704, -4.8698730, -2.7387776, -2.0318925, 2.0156393
4: -9.4584808, -6.7071004, -9.4327488, -6.7281561, -2.0857730, 2.0782294
5: -13.8432360, -11.1644611, -13.7964449, -11.1964836, -1.7362056, 1.7259204
6: -16.3860168, -12.7117920, -16.3357639, -12.7602463, -2.4343989, 2.4620488
7: -4.0861788, -1.3603723, -4.0313134, -1.3719928, -2.5637846, 2.5188150
8: -6.1121640, -3.6189246, -6.0330544, -3.6418304, -2.1258526, 2.0914578
9: -11.8736143, -9.2707195, -11.8301039, -9.3292351, -1.8722501, 1.8587072

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135085, upper bound: 0.7167889
time: 5.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7135085, upper bound: 0.7168615
time: 7.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1252794, -10.4250698, -13.1205950, -10.4745903, -1.9179733, 1.9136659
1: -7.1645441, -4.1780934, -7.1296897, -4.1796746, -2.4537754, 2.4378810
2: 9.3305330, 11.2835846, 9.3672371, 11.2968464, -1.6541388, 1.6307914
3: -4.8958139, -2.7366629, -4.8732843, -2.7360294, -2.0232515, 2.0074062
4: -9.4388762, -6.7178192, -9.4394026, -6.7191224, -2.0742455, 2.0728769
5: -13.8045349, -11.1674681, -13.8230257, -11.1769066, -1.7350197, 1.7325845
6: -16.3593140, -12.7413216, -16.3432884, -12.7533932, -2.4309511, 2.4293365
7: -4.0623298, -1.3691754, -4.0589752, -1.3445439, -2.5643420, 2.5327888
8: -6.0695133, -3.6219544, -6.0655403, -3.6194997, -2.1235843, 2.1201873
9: -11.8542051, -9.2798882, -11.8445396, -9.3180113, -1.8593183, 1.8610156

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094796, upper bound: 0.7217332
time: 9.03 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094796, upper bound: 0.7218066
time: 6.49 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1807718, -10.4064236, -13.1208735, -10.4712601, -1.9441488, 1.9292779
1: -7.1893067, -4.1606054, -7.1299486, -4.1781449, -2.4634502, 2.4719930
2: 9.3079548, 11.3110723, 9.3652439, 11.2971678, -1.6776943, 1.6479750
3: -4.9131432, -2.7291107, -4.8734827, -2.7353947, -2.0372810, 2.0242887
4: -9.4613609, -6.7055187, -9.4399672, -6.7172918, -2.0966332, 2.0875936
5: -13.8439064, -11.1540127, -13.8231468, -11.1743793, -1.7468538, 1.7437508
6: -16.3868942, -12.7092943, -16.3434467, -12.7524519, -2.4439073, 2.4714258
7: -4.0982437, -1.3592594, -4.0600042, -1.3428602, -2.5880136, 2.5430241
8: -6.1142573, -3.6081324, -6.0659218, -3.6171589, -2.1421494, 2.1331496
9: -11.8797626, -9.2700787, -11.8452406, -9.3175812, -1.8892965, 1.8711276

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140442, upper bound: 0.7217576
time: 7.68 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7140442, upper bound: 0.7218281
time: 8.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7090476, upper bound: 0.7089447
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7090720, upper bound: 0.7135107
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7140230, upper bound: 0.7094841
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7140466, upper bound: 0.7140462
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7167642, upper bound: 0.7089425
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7167895, upper bound: 0.7135080
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7217333, upper bound: 0.7094791
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7217578, upper bound: 0.7140442
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7167646
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7089426, upper bound: 0.7168382
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7135085, upper bound: 0.7167889
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7135085, upper bound: 0.7168615
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7094796, upper bound: 0.7217332
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7094796, upper bound: 0.7218066
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7140442, upper bound: 0.7217576
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.33
Output dim: 2, lower bound: -0.7140442, upper bound: 0.7218281

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.0910816, -10.4806509, -13.0917778, -10.4895096, -1.8613629, 1.8726981
1: -7.1217389, -4.2024674, -7.1239920, -4.2059631, -2.3890553, 2.3956056
2: 9.3842630, 11.2651196, 9.3836346, 11.2651958, -1.6028404, 1.6034713
3: -4.8672981, -2.7444370, -4.8677659, -2.7450662, -1.9799981, 1.9813085
4: -9.4293842, -6.7345381, -9.4308205, -6.7382183, -2.0372362, 2.0402994
5: -13.7895317, -11.2018566, -13.7898912, -11.1981735, -1.6799078, 1.6766288
6: -16.3333874, -12.7643709, -16.3338509, -12.7644615, -2.3864889, 2.3901575
7: -4.0283980, -1.3756001, -4.0382290, -1.3793485, -2.4953208, 2.5069466
8: -6.0298252, -3.6514821, -6.0310073, -3.6468611, -2.0574627, 2.0534782
9: -11.8084974, -9.3302689, -11.8129864, -9.3308334, -1.8062623, 1.8112974

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045056, upper bound: 0.7089470
time: 6.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045056, upper bound: 0.7089471
time: 7.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.0913582, -10.4773235, -13.1472139, -10.4708252, -1.8805580, 1.9071765
1: -7.1219945, -4.2009678, -7.1486449, -4.1880703, -2.4253502, 2.4206185
2: 9.3822708, 11.2654457, 9.3610516, 11.2926903, -1.6208150, 1.6312628
3: -4.8674965, -2.7438073, -4.8849335, -2.7374225, -1.9968653, 2.0010161
4: -9.4299355, -6.7327080, -9.4533319, -6.7259111, -2.0519013, 2.0662389
5: -13.7896500, -11.1993275, -13.8293238, -11.1847858, -1.6910553, 1.7056730
6: -16.3335438, -12.7634335, -16.3612938, -12.7316542, -2.4351826, 2.4163213
7: -4.0294166, -1.3739150, -4.0736127, -1.3694422, -2.5055304, 2.5453525
8: -6.0302129, -3.6491446, -6.0756149, -3.6329036, -2.0703974, 2.0943069
9: -11.8091936, -9.3298388, -11.8388500, -9.3209410, -1.8172092, 1.8413572

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045056, upper bound: 0.7134874
time: 6.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045055, upper bound: 0.7135108
time: 7.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.0974846, -10.4749250, -13.0932112, -10.4886580, -1.8693833, 1.8791211
1: -7.1282449, -4.1939859, -7.1268229, -4.2051702, -2.3957171, 2.4063077
2: 9.3691902, 11.2829676, 9.3775482, 11.2661619, -1.6138000, 1.6246815
3: -4.8709021, -2.7410612, -4.8687859, -2.7439206, -1.9886694, 1.9863806
4: -9.4365892, -6.7236662, -9.4337177, -6.7366371, -2.0466232, 2.0547686
5: -13.8162327, -11.1797600, -13.7905636, -11.1877298, -1.7113018, 1.6872382
6: -16.3410568, -12.7565727, -16.3347168, -12.7619619, -2.3981662, 2.4013300
7: -4.0570655, -1.3464608, -4.0503278, -1.3782473, -2.5194979, 2.5493529
8: -6.0626988, -3.6268106, -6.0331621, -3.6360550, -2.0996108, 2.0697317
9: -11.8236160, -9.3186131, -11.8191757, -9.3301964, -1.8186512, 1.8283739

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094817, upper bound: 0.7094821
time: 7.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094817, upper bound: 0.7094816
time: 6.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.0977612, -10.4715939, -13.1486292, -10.4699678, -1.8885727, 1.9124104
1: -7.1285019, -4.1924849, -7.1514740, -4.1872725, -2.4320168, 2.4313273
2: 9.3671961, 11.2832899, 9.3549786, 11.2936516, -1.6307602, 1.6484125
3: -4.8710999, -2.7404275, -4.8859529, -2.7362618, -2.0055246, 2.0060940
4: -9.4371519, -6.7218351, -9.4562130, -6.7243214, -2.0612574, 2.0806637
5: -13.8163548, -11.1772299, -13.8299961, -11.1743345, -1.7224884, 1.7163239
6: -16.3412132, -12.7556334, -16.3621674, -12.7291603, -2.4468160, 2.4258347
7: -4.0580950, -1.3447757, -4.0856752, -1.3683286, -2.5297408, 2.5731866
8: -6.0630836, -3.6244698, -6.0777597, -3.6221037, -2.1126261, 2.1106114
9: -11.8243189, -9.3181820, -11.8449917, -9.3202982, -1.8296001, 1.8583875

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094817, upper bound: 0.7140237
time: 7.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7094816, upper bound: 0.7140471
time: 7.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.0910816, -10.4806509, -13.1238499, -10.4259262, -1.8803384, 1.9087615
1: -7.1217389, -4.2024674, -7.1615548, -4.1788974, -2.4128857, 2.4279442
2: 9.3842630, 11.2651196, 9.3370323, 11.2826071, -1.6224432, 1.6258252
3: -4.8672981, -2.7444370, -4.8947158, -2.7378223, -1.9881735, 2.0100088
4: -9.4293842, -6.7345381, -9.4359770, -6.7200317, -2.0578256, 2.0460844
5: -13.7895317, -11.2018566, -13.8038654, -11.1786776, -1.6988115, 1.6905043
6: -16.3333874, -12.7643709, -16.3582268, -12.7438364, -2.4082499, 2.4153452
7: -4.0283980, -1.3756001, -4.0502133, -1.3702836, -2.5057592, 2.5201621
8: -6.0298252, -3.6514821, -6.0673428, -3.6328115, -2.0720568, 2.0908246
9: -11.8084974, -9.3302689, -11.8478231, -9.2805243, -1.8270655, 1.8477898

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7122252, upper bound: 0.7089420
time: 10.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7122252, upper bound: 0.7089446
time: 5.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.0913582, -10.4773235, -13.1793385, -10.4072809, -1.8959472, 1.9349036
1: -7.1219945, -4.2009678, -7.1863179, -4.1614141, -2.4493275, 2.4376154
2: 9.3822708, 11.2654457, 9.3144436, 11.3101082, -1.6352842, 1.6493673
3: -4.8674965, -2.7438073, -4.9120536, -2.7302830, -2.0049782, 2.0237870
4: -9.4299355, -6.7327080, -9.4584770, -6.7077332, -2.0725675, 2.0719886
5: -13.7896500, -11.1993275, -13.8432341, -11.1652203, -1.7100382, 1.7177008
6: -16.3335438, -12.7634335, -16.3858032, -12.7118120, -2.4568324, 2.4289980
7: -4.0294166, -1.3739150, -4.0861692, -1.3603802, -2.5159636, 2.5587797
8: -6.0302129, -3.6491446, -6.1121492, -3.6189837, -2.0849357, 2.1140323
9: -11.8091936, -9.3298388, -11.8734369, -9.2707186, -1.8371739, 1.8778210

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7122252, upper bound: 0.7134866
time: 7.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7122251, upper bound: 0.7135111
time: 4.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.0974846, -10.4749250, -13.1252604, -10.4250708, -1.8883727, 1.9151628
1: -7.1282449, -4.1939859, -7.1643925, -4.1781006, -2.4195576, 2.4346595
2: 9.3691902, 11.2829676, 9.3309612, 11.2835732, -1.6334009, 1.6383963
3: -4.8709021, -2.7410612, -4.8957481, -2.7366769, -1.9968476, 2.0150914
4: -9.4365892, -6.7236662, -9.4388733, -6.7184553, -2.0672135, 2.0605531
5: -13.8162327, -11.1797600, -13.8045340, -11.1682301, -1.7170310, 1.7011096
6: -16.3410568, -12.7565727, -16.3591003, -12.7413387, -2.4199333, 2.4255397
7: -4.0570655, -1.3464608, -4.0623174, -1.3691845, -2.5299330, 2.5614400
8: -6.0626988, -3.6268106, -6.0694962, -3.6220140, -2.1119637, 2.1070738
9: -11.8236160, -9.3186131, -11.8540192, -9.2798891, -1.8394790, 1.8648732

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7171929, upper bound: 0.7094796
time: 7.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7171929, upper bound: 0.7094791
time: 7.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.0977612, -10.4715939, -13.1807556, -10.4064236, -1.9039860, 1.9401140
1: -7.1285019, -4.1924849, -7.1891546, -4.1606121, -2.4560013, 2.4443412
2: 9.3671961, 11.2832899, 9.3083820, 11.3110657, -1.6452279, 1.6619496
3: -4.8710999, -2.7404275, -4.9130864, -2.7291231, -2.0136342, 2.0291712
4: -9.4371519, -6.7218351, -9.4613590, -6.7061505, -2.0819297, 2.0864134
5: -13.8163548, -11.1772299, -13.8439064, -11.1547718, -1.7281981, 1.7283485
6: -16.3412132, -12.7556334, -16.3866844, -12.7093134, -2.4682212, 2.4385014
7: -4.0580950, -1.3447757, -4.0982351, -1.3592682, -2.5401721, 2.5851102
8: -6.0630836, -3.6244698, -6.1142416, -3.6081924, -2.1249352, 2.1303368
9: -11.8243189, -9.3181820, -11.8795843, -9.2700777, -1.8495927, 1.8911669

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7171929, upper bound: 0.7140231
time: 5.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7171929, upper bound: 0.7140441
time: 8.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1238499, -10.4259262, -13.0910816, -10.4806509, -1.9087620, 1.8803383
1: -7.1615548, -4.1788974, -7.1217389, -4.2024674, -2.4279442, 2.4128847
2: 9.3370323, 11.2826071, 9.3842630, 11.2651196, -1.6258254, 1.6224432
3: -4.8947158, -2.7378223, -4.8672981, -2.7444370, -2.0100088, 1.9881735
4: -9.4359770, -6.7200317, -9.4293842, -6.7345381, -2.0460839, 2.0578256
5: -13.8038654, -11.1786776, -13.7895317, -11.2018566, -1.6905046, 1.6988115
6: -16.3582268, -12.7438364, -16.3333874, -12.7643709, -2.4153452, 2.4082499
7: -4.0502133, -1.3702836, -4.0283980, -1.3756001, -2.5201612, 2.5057583
8: -6.0673428, -3.6328115, -6.0298252, -3.6514821, -2.0908251, 2.0720568
9: -11.8478231, -9.2805243, -11.8084974, -9.3302689, -1.8477898, 1.8270657

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045031, upper bound: 0.7167638
time: 8.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045031, upper bound: 0.7167639
time: 5.30 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1238785, -10.4259253, -13.1232367, -10.4170876, -1.9147601, 1.9034402
1: -7.1617966, -4.1788874, -7.1595449, -4.1753893, -2.4422770, 2.4357634
2: 9.3363457, 11.2826214, 9.3369970, 11.2825546, -1.6324444, 1.6318467
3: -4.8948231, -2.7378030, -4.8943481, -2.7371776, -2.0102520, 2.0089202
4: -9.4359818, -6.7190094, -9.4345541, -6.7153358, -2.0666208, 2.0635409
5: -13.8038654, -11.1774511, -13.8035259, -11.1811695, -1.7268724, 1.7275221
6: -16.3585701, -12.7438087, -16.3581104, -12.7437544, -2.4384446, 2.4347887
7: -4.0502357, -1.3702722, -4.0404205, -1.3665221, -2.5311236, 2.5195074
8: -6.0673676, -3.6327143, -6.0662184, -3.6373367, -2.1146173, 2.1160460
9: -11.8481216, -9.2805252, -11.8436480, -9.2799835, -1.8512139, 1.8461909

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045031, upper bound: 0.7168382
time: 6.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7045031, upper bound: 0.7168381
time: 5.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1793385, -10.4072809, -13.0913582, -10.4773235, -1.9349036, 1.8959472
1: -7.1863179, -4.1614141, -7.1219945, -4.2009678, -2.4376156, 2.4493275
2: 9.3144436, 11.3101082, 9.3822708, 11.2654457, -1.6493669, 1.6352842
3: -4.9120536, -2.7302830, -4.8674965, -2.7438073, -2.0237870, 2.0049782
4: -9.4584770, -6.7077332, -9.4299355, -6.7327080, -2.0719886, 2.0725679
5: -13.8432341, -11.1652203, -13.7896500, -11.1993275, -1.7177007, 1.7100384
6: -16.3858032, -12.7118120, -16.3335438, -12.7634335, -2.4289985, 2.4568324
7: -4.0861692, -1.3603802, -4.0294166, -1.3739150, -2.5587797, 2.5159636
8: -6.1121492, -3.6189837, -6.0302129, -3.6491446, -2.1140323, 2.0849361
9: -11.8734369, -9.2707186, -11.8091936, -9.3298388, -1.8778210, 1.8371737

Time for backsubstitution: 14.50 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.6408071517944336
rel_dist={2: [-0.7218570837425542, 0.7218564741351408]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117883, upper bound: 0.6060459
time: 6.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118731, upper bound: 0.6118744
time: 6.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.01
Output dim: 2, lower bound: -0.6117883, upper bound: 0.6060459
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.01
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

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6111854, upper bound: 0.6021157
time: 9.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117819, upper bound: 0.6060391
time: 6.04 seconds

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

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6198
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6198

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112706, upper bound: 0.6079467
time: 8.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118666, upper bound: 0.6118675
time: 13.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 36.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.61
Output dim: 2, lower bound: -0.6111854, upper bound: 0.6021157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.61
Output dim: 2, lower bound: -0.6117819, upper bound: 0.6060391
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.61
Output dim: 2, lower bound: -0.6112706, upper bound: 0.6079467
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.61
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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6077343, upper bound: 0.6020745
time: 6.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6111817, upper bound: 0.6021125
time: 7.58 seconds

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

Time for backsubstitution: 13.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6083403, upper bound: 0.6060015
time: 9.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6117781, upper bound: 0.6060361
time: 6.93 seconds

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

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6078210, upper bound: 0.6079031
time: 7.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6112669, upper bound: 0.6079410
time: 5.60 seconds

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

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6084251, upper bound: 0.6118273
time: 8.20 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118629, upper bound: 0.6118643
time: 6.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.60 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6077343, upper bound: 0.6020745
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6111817, upper bound: 0.6021125
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6083403, upper bound: 0.6060015
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6117781, upper bound: 0.6060361
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6078210, upper bound: 0.6079031
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6112669, upper bound: 0.6079410
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6084251, upper bound: 0.6118273
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 2, lower bound: -0.6118629, upper bound: 0.6118643

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1469107, -10.4710131, -13.1036415, -10.4771442, -1.8008904, 1.7856731
1: -7.1480312, -4.1882458, -7.1227813, -4.1934977, -2.3306932, 2.3272052
2: 9.3623676, 11.2924824, 9.3812237, 11.2728167, -1.5621996, 1.5464177
3: -4.8847141, -2.7376733, -4.8687730, -2.7411387, -1.9244757, 1.9172277
4: -9.4527082, -6.7262559, -9.4314241, -6.7302308, -1.9672079, 1.9515524
5: -13.8291779, -11.1870461, -13.7932148, -11.1977844, -1.6096158, 1.5964246
6: -16.3611069, -12.7321978, -16.3347473, -12.7617588, -2.2948637, 2.3124530
7: -4.0710020, -1.3696866, -4.0304308, -1.3728867, -2.4662285, 2.4280891
8: -6.0751581, -3.6352429, -6.0317430, -3.6452703, -2.0260425, 1.9998302
9: -11.8375225, -9.3210793, -11.8202581, -9.3295183, -1.7354627, 1.7234492

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054405, upper bound: 0.6021123
time: 7.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054405, upper bound: 0.6021129
time: 8.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1486273, -10.4699669, -13.1100397, -10.4714203, -1.8065100, 1.7938187
1: -7.1514726, -4.1872725, -7.1292934, -4.1850166, -2.3420172, 2.3339181
2: 9.3549795, 11.2936506, 9.3661432, 11.2906618, -1.5786078, 1.5555456
3: -4.8859525, -2.7362628, -4.8723788, -2.7377582, -1.9299259, 1.9261508
4: -9.4562130, -6.7243204, -9.4386396, -6.7193632, -1.9824319, 1.9611425
5: -13.8299961, -11.1743374, -13.8199167, -11.1756811, -1.6186197, 1.6265557
6: -16.3621674, -12.7291622, -16.3424244, -12.7539616, -2.3043964, 2.3233802
7: -4.0856719, -1.3683290, -4.0591145, -1.3437498, -2.4950929, 2.4509134
8: -6.0777597, -3.6221075, -6.0646124, -3.6205983, -2.0411983, 2.0418305
9: -11.8449917, -9.3202991, -11.8353920, -9.3178635, -1.7538936, 1.7354383

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060378, upper bound: 0.6060363
time: 9.43 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6060378, upper bound: 0.6060358
time: 7.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1790466, -10.4074688, -13.1144705, -10.4769888, -1.8281476, 1.8119769
1: -7.1858311, -4.1615839, -7.1234312, -4.1866078, -2.3541689, 2.3593147
2: 9.3153925, 11.3099041, 9.3803339, 11.2793217, -1.5869980, 1.5602517
3: -4.9118814, -2.7305236, -4.8698711, -2.7387781, -1.9492459, 1.9330664
4: -9.4578590, -6.7075443, -9.4327478, -6.7281580, -1.9818165, 1.9750643
5: -13.8430882, -11.1668396, -13.7964439, -11.1964855, -1.6321445, 1.6218550
6: -16.3857918, -12.7123327, -16.3357620, -12.7602491, -2.3103824, 2.3366938
7: -4.0835643, -1.3606172, -4.0313129, -1.3719947, -2.4820495, 2.4398537
8: -6.1117053, -3.6212697, -6.0330524, -3.6418352, -2.0524907, 2.0174527
9: -11.8722591, -9.2708607, -11.8300972, -9.3292351, -1.7636175, 1.7523744

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054386, upper bound: 0.6078549
time: 7.20 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6054386, upper bound: 0.6078555
time: 13.27 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1252728, -10.4250708, -13.1204090, -10.4766903, -1.8061531, 1.8066921
1: -7.1645188, -4.1780939, -7.1295290, -4.1806393, -2.3499331, 2.3351693
2: 9.3305998, 11.2835827, 9.3684921, 11.2966385, -1.5781784, 1.5518272
3: -4.8958044, -2.7366652, -4.8731565, -2.7364302, -1.9417307, 1.9253583
4: -9.4388762, -6.7179165, -9.4390488, -6.7202792, -1.9706607, 1.9699693
5: -13.8045359, -11.1675892, -13.8229485, -11.1785021, -1.6275668, 1.6306834
6: -16.3592815, -12.7413254, -16.3431892, -12.7539902, -2.3058372, 2.3068032
7: -4.0623255, -1.3691773, -4.0583220, -1.3456068, -2.4836688, 2.4522433
8: -6.0695100, -3.6219664, -6.0652947, -3.6209774, -2.0474949, 2.0473914
9: -11.8541756, -9.2798882, -11.8440866, -9.3182793, -1.7518203, 1.7539387

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6084236
time: 7.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6118274
time: 9.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1807690, -10.4064236, -13.1208658, -10.4712601, -1.8337417, 1.8201530
1: -7.1892829, -4.1606064, -7.1299467, -4.1781483, -2.3614595, 2.3660331
2: 9.3080215, 11.3110704, 9.3652458, 11.2971649, -1.6007905, 1.5693793
3: -4.9131346, -2.7291124, -4.8734818, -2.7353954, -1.9549789, 1.9419804
4: -9.4613590, -6.7056131, -9.4399643, -6.7172956, -1.9928966, 1.9846625
5: -13.8439054, -11.1541328, -13.8231440, -11.1743813, -1.6411457, 1.6408221
6: -16.3868599, -12.7092991, -16.3434467, -12.7524509, -2.3199060, 2.3467278
7: -4.0982404, -1.3592625, -4.0600033, -1.3428612, -2.5085039, 2.4626741
8: -6.1142540, -3.6081443, -6.0659204, -3.6171632, -2.0676408, 2.0594797
9: -11.8797350, -9.2700787, -11.8452349, -9.3175831, -1.7820575, 1.7643933

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060359, upper bound: 0.6117769
time: 9.96 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6060359, upper bound: 0.6117804
time: 6.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.40 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6054405, upper bound: 0.6021123
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6054405, upper bound: 0.6021129
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6060378, upper bound: 0.6060363
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6060378, upper bound: 0.6060358
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6054386, upper bound: 0.6078549
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6054386, upper bound: 0.6078555
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6084236
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6084252, upper bound: 0.6118274
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6060359, upper bound: 0.6117769
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.40
Output dim: 2, lower bound: -0.6060359, upper bound: 0.6117804

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1252728, -10.4250708, -13.1751652, -10.4656677, -1.8159375, 1.8219202
1: -7.1645188, -4.1780939, -7.1536036, -4.1664457, -2.3582556, 2.3497739
2: 9.3305998, 11.2835827, 9.3505325, 11.3232861, -1.5819416, 1.5710516
3: -4.8958044, -2.7366652, -4.8898377, -2.7302446, -1.9474156, 1.9426301
4: -9.4388762, -6.7179165, -9.4602356, -6.7121940, -1.9789982, 1.9872515
5: -13.8045359, -11.1675892, -13.8618031, -11.1709146, -1.6309733, 1.6391203
6: -16.3592815, -12.7413254, -16.3702698, -12.7234707, -2.3372269, 2.3206229
7: -4.0623255, -1.3691773, -4.0911603, -1.3395853, -2.4872131, 2.4853706
8: -6.0695100, -3.6219664, -6.1087465, -3.6124783, -2.0531673, 2.0619068
9: -11.8541756, -9.2798882, -11.8681593, -9.3093519, -1.7604413, 1.7755995

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6044946, upper bound: 0.6112290
time: 8.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6044946, upper bound: 0.6087337
time: 5.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1807489, -10.4064236, -13.0977612, -10.4715977, -1.8301814, 1.7948613
1: -7.1891227, -4.1606126, -7.1285014, -4.1924858, -2.3423469, 2.3549228
2: 9.3084726, 11.3110619, 9.3671989, 11.2832890, -1.5850239, 1.5666232
3: -4.9130745, -2.7291279, -4.8710995, -2.7404280, -1.9468653, 1.9324112
4: -9.4613571, -6.7062864, -9.4371519, -6.7218380, -1.9847755, 1.9789672
5: -13.8439054, -11.1549387, -13.8163509, -11.1772327, -1.6241410, 1.6252509
6: -16.3866386, -12.7093201, -16.3412170, -12.7556372, -2.3144846, 2.3439231
7: -4.0982304, -1.3592696, -4.0580940, -1.3447776, -2.5057216, 2.4598217
8: -6.1142354, -3.6082072, -6.0630836, -3.6244717, -2.0562382, 2.0503969
9: -11.8795462, -9.2700787, -11.8243179, -9.3181829, -1.7826133, 1.7428586

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6111808
time: 7.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6086835
time: 7.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.1807804, -10.4064207, -13.1298771, -10.4080362, -1.8368030, 1.8188455
1: -7.1893973, -4.1606011, -7.1663308, -4.1654010, -2.3671615, 2.3706121
2: 9.3076973, 11.3110800, 9.3199673, 11.3007193, -1.5979867, 1.5809848
3: -4.9131775, -2.7291019, -4.8981895, -2.7331676, -1.9534340, 1.9521513
4: -9.4613638, -6.7051344, -9.4423285, -6.7026458, -1.9996877, 1.9839911
5: -13.8439035, -11.1535530, -13.8303528, -11.1565514, -1.6436446, 1.6505839
6: -16.3870220, -12.7092857, -16.3659859, -12.7350569, -2.3379068, 2.3582790
7: -4.0982461, -1.3592548, -4.0701399, -1.3357067, -2.5135555, 2.4734473
8: -6.1142673, -3.6080999, -6.0994401, -3.6103830, -2.0749369, 2.0775223
9: -11.8798704, -9.2700796, -11.8594818, -9.2678986, -1.7880793, 1.7617188

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6198
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6198

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6112692
time: 4.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6086843
time: 5.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.92 seconds
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.92
Output dim: 2, lower bound: -0.6044946, upper bound: 0.6112290
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.92
Output dim: 2, lower bound: -0.6044946, upper bound: 0.6087337
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.92
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6111808
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 24.92
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6086835
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.92
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6112692
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.92
Output dim: 2, lower bound: -0.6021122, upper bound: 0.6086843

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1223717, -10.4268551, -13.1751652, -10.4656677, -1.8121142, 1.8197473
1: -7.1586676, -4.1797562, -7.1536036, -4.1664457, -2.3523638, 2.3459220
2: 9.3431339, 11.2815685, 9.3505325, 11.3232861, -1.5688524, 1.5721407
3: -4.8936949, -2.7390218, -4.8898377, -2.7302446, -1.9432366, 1.9361973
4: -9.4329109, -6.7212029, -9.4602356, -6.7121940, -1.9712396, 1.9816542
5: -13.8031454, -11.1891451, -13.8618031, -11.1709146, -1.6285591, 1.6173695
6: -16.3574772, -12.7464647, -16.3702698, -12.7234707, -2.3315296, 2.3136961
7: -4.0373592, -1.3714747, -4.0911603, -1.3395853, -2.4589243, 2.4774604
8: -6.0650463, -3.6442475, -6.1087465, -3.6124783, -2.0465765, 2.0391769
9: -11.8414135, -9.2812166, -11.8681593, -9.3093519, -1.7470307, 1.7726479

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986649, upper bound: 0.6111424
time: 9.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986649, upper bound: 0.6111457
time: 5.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1778374, -10.4082146, -13.0977612, -10.4715977, -1.8263857, 1.7926567
1: -7.1832714, -4.1622834, -7.1285014, -4.1924858, -2.3364360, 2.3540049
2: 9.3209867, 11.3090677, 9.3671989, 11.2832890, -1.5719099, 1.5629148
3: -4.9109631, -2.7315109, -4.8710995, -2.7404280, -1.9426754, 1.9258637
4: -9.4554214, -6.7095861, -9.4371519, -6.7218380, -1.9771051, 1.9766126
5: -13.8425112, -11.1765060, -13.8163509, -11.1772327, -1.6217282, 1.6034952
6: -16.3848190, -12.7144594, -16.3412170, -12.7556372, -2.3088017, 2.3369820
7: -4.0733395, -1.3615899, -4.0580940, -1.3447776, -2.4773941, 2.4645414
8: -6.1098828, -3.6304798, -6.0630836, -3.6244717, -2.0496459, 2.0276599
9: -11.8668861, -9.2714157, -11.8243179, -9.3181829, -1.7692237, 1.7398772

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6020739, upper bound: 0.6077335
time: 7.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6020744, upper bound: 0.6077354
time: 5.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1778708, -10.4082127, -13.1298771, -10.4080362, -1.8330073, 1.8175285
1: -7.1835475, -4.1622734, -7.1663308, -4.1654010, -2.3612540, 2.3667471
2: 9.3202095, 11.3090849, 9.3199673, 11.3007193, -1.5848722, 1.5772686
3: -4.9110675, -2.7314868, -4.8981895, -2.7331676, -1.9499388, 1.9456344
4: -9.4554281, -6.7084341, -9.4423285, -6.7026458, -1.9919171, 1.9816298
5: -13.8425131, -11.1751194, -13.8303528, -11.1565514, -1.6412313, 1.6288289
6: -16.3852100, -12.7144232, -16.3659859, -12.7350569, -2.3322210, 2.3513367
7: -4.0733585, -1.3615761, -4.0701399, -1.3357067, -2.4852281, 2.4781685
8: -6.1099124, -3.6303697, -6.0994401, -3.6103830, -2.0683646, 2.0547857
9: -11.8672085, -9.2714167, -11.8594818, -9.2678986, -1.7746897, 1.7647364

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6021455, upper bound: 0.6098890
time: 15.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6022893, upper bound: 0.6112664
time: 10.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 40.62 seconds
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 40.62
Output dim: 2, lower bound: -0.5986649, upper bound: 0.6111424
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 40.62
Output dim: 2, lower bound: -0.5986649, upper bound: 0.6111457
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 40.62
Output dim: 2, lower bound: -0.6020739, upper bound: 0.6077335
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 40.62
Output dim: 2, lower bound: -0.6020744, upper bound: 0.6077354
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 40.62
Output dim: 2, lower bound: -0.6021455, upper bound: 0.6098890
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 40.62
Output dim: 2, lower bound: -0.6022893, upper bound: 0.6112664

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1223555, -10.4268551, -13.1521149, -10.4660320, -1.8088670, 1.7944105
1: -7.1585069, -4.1797628, -7.1521354, -4.1806097, -2.3332839, 2.3376870
2: 9.3435879, 11.2815619, 9.3525257, 11.3094225, -1.5530930, 1.5693867
3: -4.8936238, -2.7390358, -4.8874426, -2.7352698, -1.9351084, 1.9267721
4: -9.4329071, -6.7218790, -9.4574394, -6.7167521, -1.9582748, 1.9762862
5: -13.8031416, -11.1899529, -13.8550501, -11.1738119, -1.6115866, 1.6017699
6: -16.3572521, -12.7464838, -16.3680153, -12.7265844, -2.3261127, 2.3109462
7: -4.0373445, -1.3714857, -4.0893116, -1.3415117, -2.4561353, 2.4745829
8: -6.0650291, -3.6443119, -6.1058922, -3.6197739, -2.0351982, 2.0301869
9: -11.8412180, -9.2812138, -11.8473377, -9.3099804, -1.7531233, 1.7512066

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5972730, upper bound: 0.6109984
time: 11.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986637, upper bound: 0.6111410
time: 9.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1223879, -10.4268541, -13.1842260, -10.4024887, -1.8152270, 1.8218493
1: -7.1587815, -4.1797519, -7.1900635, -4.1539421, -2.3533163, 2.3504770
2: 9.3428116, 11.2815762, 9.3052368, 11.3268423, -1.5662680, 1.5835230
3: -4.8937449, -2.7390106, -4.9146748, -2.7281084, -1.9354672, 1.9464154
4: -9.4329128, -6.7207203, -9.4625931, -6.6975641, -1.9779825, 1.9834859
5: -13.8031425, -11.1885681, -13.8689575, -11.1530247, -1.6310279, 1.6270000
6: -16.3576355, -12.7464504, -16.3928909, -12.7067232, -2.3490901, 2.3251913
7: -4.0373697, -1.3714695, -4.1018977, -1.3324382, -2.4639463, 2.4869056
8: -6.0650578, -3.6442022, -6.1424022, -3.6058040, -2.0538898, 2.0571885
9: -11.8415546, -9.2812157, -11.8822069, -9.2597523, -1.7559624, 1.7790864

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5972729, upper bound: 0.6110853
time: 5.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5986637, upper bound: 0.6112286
time: 8.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1766758, -10.4134922, -13.1208534, -10.4185057, -1.8213656, 1.8026252
1: -7.1822143, -4.1670074, -7.1578994, -4.1746573, -2.3499489, 2.3522770
2: 9.3246346, 11.3078117, 9.3293915, 11.2930393, -1.5713260, 1.5661514
3: -4.9100275, -2.7341571, -4.8920050, -2.7382140, -1.9437199, 1.9371057
4: -9.4530430, -6.7122159, -9.4342251, -6.7102575, -1.9794009, 1.9633784
5: -13.8406458, -11.1767635, -13.8238125, -11.1602192, -1.6356537, 1.6212343
6: -16.3841877, -12.7190552, -16.3559494, -12.7472868, -2.3186071, 2.3374052
7: -4.0691996, -1.3620763, -4.0602994, -1.3375697, -2.4763675, 2.4652257
8: -6.1086502, -3.6362267, -6.0896215, -3.6220317, -2.0548797, 2.0360913
9: -11.8604832, -9.2715836, -11.8440475, -9.2731209, -1.7611804, 1.7493138

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6012214, upper bound: 0.6033231
time: 7.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6012214, upper bound: 0.6067663
time: 7.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1778717, -10.4082174, -13.1298733, -10.4080458, -1.8230410, 1.8152089
1: -7.1835446, -4.1622777, -7.1663294, -4.1654129, -2.3544641, 2.3616161
2: 9.3202114, 11.3090849, 9.3199720, 11.3007154, -1.5811779, 1.5690796
3: -4.9110656, -2.7314897, -4.8981862, -2.7331715, -1.9483094, 1.9440351
4: -9.4554262, -6.7084374, -9.4423218, -6.7026525, -1.9857731, 1.9810109
5: -13.8425112, -11.1751175, -13.8303471, -11.1565542, -1.6388307, 1.6273134
6: -16.3852100, -12.7144260, -16.3659821, -12.7350616, -2.3242104, 2.3464189
7: -4.0733557, -1.3615761, -4.0701308, -1.3357062, -2.4854827, 2.4778214
8: -6.1099105, -3.6303759, -6.0994382, -3.6103964, -2.0585399, 2.0494115
9: -11.8672028, -9.2714167, -11.8594694, -9.2678995, -1.7700076, 1.7570665

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4586
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6009112, upper bound: 0.6111255
time: 5.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6009111, upper bound: 0.6112684
time: 5.49 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 25.16 seconds
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.5972730, upper bound: 0.6109984
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.5986637, upper bound: 0.6111410
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.5972729, upper bound: 0.6110853
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.5986637, upper bound: 0.6112286
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.6012214, upper bound: 0.6033231
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.6012214, upper bound: 0.6067663
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.6009112, upper bound: 0.6111255
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.16
Output dim: 2, lower bound: -0.6009111, upper bound: 0.6112684

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1133366, -10.4373140, -13.1509113, -10.4713364, -1.7930553, 1.7827637
1: -7.1500845, -4.1889668, -7.1508360, -4.1854944, -2.3185499, 2.3265219
2: 9.3530245, 11.2738495, 9.3569441, 11.3081903, -1.5419779, 1.5557606
3: -4.8874416, -2.7440577, -4.8863869, -2.7379467, -1.9265361, 1.9205689
4: -9.4248142, -6.7294850, -9.4551239, -6.7205515, -1.9401422, 1.9638076
5: -13.7966042, -11.1936131, -13.8530836, -11.1754436, -1.6040111, 1.5961740
6: -16.3471947, -12.7587099, -16.3670540, -12.7312689, -2.3120108, 2.2974777
7: -4.0276427, -1.3733544, -4.0850554, -1.3420084, -2.4433584, 2.4656396
8: -6.0551577, -3.6559653, -6.1044559, -3.6256590, -2.0164289, 2.0165317
9: -11.8257942, -9.2864447, -11.8405180, -9.3101444, -1.7375522, 1.7375677

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5972730, upper bound: 0.6097645
time: 12.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5972730, upper bound: 0.6109984
time: 8.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1223507, -10.4268646, -13.1521120, -10.4660416, -1.8038704, 1.7844815
1: -7.1585045, -4.1797738, -7.1521235, -4.1807375, -2.3279581, 2.3312092
2: 9.3435917, 11.2815590, 9.3525276, 11.3093662, -1.5450234, 1.5656731
3: -4.8936224, -2.7390382, -4.8874154, -2.7352729, -1.9316840, 1.9265289
4: -9.4329014, -6.7218847, -9.4574366, -6.7167664, -1.9577022, 1.9703934
5: -13.8031406, -11.1899586, -13.8548546, -11.1738186, -1.6100507, 1.5993192
6: -16.3572483, -12.7464914, -16.3679962, -12.7266054, -2.3211613, 2.3030691
7: -4.0373378, -1.3714852, -4.0892582, -1.3415151, -2.4553518, 2.4748030
8: -6.0650277, -3.6443243, -6.1057043, -3.6197820, -2.0298009, 2.0202160
9: -11.8412046, -9.2812166, -11.8472834, -9.3099804, -1.7443323, 1.7464988

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4586
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4586

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5985110, upper bound: 0.6097611
time: 8.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5985109, upper bound: 0.6097640
time: 10.76 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 33.42 seconds
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 33.42
Output dim: 2, lower bound: -0.5972730, upper bound: 0.6097645
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 33.42
Output dim: 2, lower bound: -0.5972730, upper bound: 0.6109984
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 33.42
Output dim: 2, lower bound: -0.5985110, upper bound: 0.6097611
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 33.42
Output dim: 2, lower bound: -0.5985109, upper bound: 0.6097640
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 33.42
Output dim: 2, lower bound: -0.5972729, upper bound: 0.6110853
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 33.42
Output dim: 2, lower bound: -0.5986637, upper bound: 0.6112286
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 33.42
Output dim: 2, lower bound: -0.6009112, upper bound: 0.6111255
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 33.42
Output dim: 2, lower bound: -0.6009111, upper bound: 0.6112684
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.56638765335083
rel_dist={2: [-0.6118841057361983, 0.6118843662850555]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2425.86 seconds
