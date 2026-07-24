## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.44380584438
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.6581917, 1.6581917)
1: (-2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.9685266, 1.9685266)
2: (-4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.8327081, 1.8327081)
3: (-12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.6218438, 2.6218441)
4: (-6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.7623563, 1.7623563)
5: (-2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.8194118, 1.8194118)
6: (2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445)
7: (-10.2777958, -8.1927624, -10.2777958, -8.1927624, -2.0850334, 2.0850334)
8: (-1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.4304194, 2.4304194)
9: (-8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.5183010, 1.5183010)

## BASE Result
execution time: IAR + LP analysis = 13.20 + 31.92 = 45.12 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.88 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.568244457244873
rel_dist={6: [-0.8170414609587353, 0.8170398270121066]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.525653600692749
rel_dist={6: [-0.6052081522674859, 0.6052060697814263]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.4518237113952637
rel_dist={6: [-0.44793373691976335, 0.4479324567017895]}

## Binary Search Result
Binary search time: 142.13 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 3412.75 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161732, upper bound: 0.7947355
time: 4.25 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8170167, upper bound: 0.8170155
time: 4.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.77 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.77
Output dim: 6, lower bound: -0.8161732, upper bound: 0.7947355
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.77
Output dim: 6, lower bound: -0.8170167, upper bound: 0.8170155

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8081179, -3.0685711, -1.3397648, 1.3445597
1: -2.8572679, -0.9035015, -2.8646710, -0.8994924, -1.6148124, 1.6198747
2: -4.1984229, -2.3881874, -4.2115669, -2.3838022, -1.6279874, 1.6361532
3: -12.6284370, -9.9407396, -12.6460953, -9.9388285, -2.1406522, 2.1540146
4: -6.0398083, -4.3091712, -6.0459118, -4.2905464, -1.5333223, 1.5234058
5: -2.8217437, -1.0503111, -2.8266516, -1.0176489, -1.5716858, 1.5434515
6: 2.2715242, 3.7956080, 2.2648110, 3.8234255, -1.5519013, 1.5307970
7: -10.2367878, -8.1965809, -10.2687864, -8.1935301, -1.8192914, 1.8497384
8: -1.9051766, 0.7185178, -1.9138527, 0.7265098, -2.0927000, 2.0945799
9: -8.4931936, -6.9887705, -8.5012283, -6.9860821, -1.4470940, 1.4480226

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947327, upper bound: 0.7947334
time: 4.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947352, upper bound: 0.7947336
time: 5.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8120556, -3.0521393, -4.8109632, -3.0668535, -1.3544354, 1.3694029
1: -2.8732462, -0.8965058, -2.8668056, -0.8982877, -1.6315322, 1.6314828
2: -4.2175093, -2.3583987, -4.2153492, -2.3826566, -1.6460826, 1.6697516
3: -12.6534863, -9.9009495, -12.6510677, -9.9383640, -2.1691957, 2.1924596
4: -6.0748878, -4.2827406, -6.0474730, -4.2851329, -1.5738578, 1.5484612
5: -2.9003108, -1.0070014, -2.8278708, -1.0084844, -1.6079705, 1.5810835
6: 2.2075305, 3.8330717, 2.2630289, 3.8312535, -1.6237230, 1.5700428
7: -10.2802534, -8.1249027, -10.2777758, -8.1927614, -1.8530948, 1.8768764
8: -1.9250176, 0.7302978, -1.9165072, 0.7287629, -2.1165066, 2.1091332
9: -8.5070438, -6.9828806, -8.5035839, -6.9852867, -1.4700322, 1.4580545

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947352, upper bound: 0.8161718
time: 4.25 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947352, upper bound: 0.8170170
time: 3.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.90
Output dim: 6, lower bound: -0.7947327, upper bound: 0.7947334
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.90
Output dim: 6, lower bound: -0.7947352, upper bound: 0.7947336
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.90
Output dim: 6, lower bound: -0.7947352, upper bound: 0.8161718
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.90
Output dim: 6, lower bound: -0.7947352, upper bound: 0.8170170

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.7981524, -3.0748608, -1.3343859, 1.3343859
1: -2.8572679, -0.9035015, -2.8572679, -0.9035015, -1.6097560, 1.6097559
2: -4.1984229, -2.3881874, -4.1984229, -2.3881874, -1.6230712, 1.6230711
3: -12.6284370, -9.9407396, -12.6284370, -9.9407396, -2.1347406, 2.1347406
4: -6.0398083, -4.3091712, -6.0398083, -4.3091712, -1.5162749, 1.5162752
5: -2.8217437, -1.0503111, -2.8217437, -1.0503111, -1.5386319, 1.5386319
6: 2.2715242, 3.7956080, 2.2715242, 3.7956080, -1.5240839, 1.5240839
7: -10.2367878, -8.1965809, -10.2367878, -8.1965809, -1.8178806, 1.8178804
8: -1.9051766, 0.7185178, -1.9051766, 0.7185178, -2.0847797, 2.0847797
9: -8.4931936, -6.9887705, -8.4931936, -6.9887705, -1.4381876, 1.4381881

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.7947142
time: 4.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947109, upper bound: 0.7947137
time: 4.04 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8119059, -3.0521817, -1.3562965, 1.3482549
1: -2.8572679, -0.9035015, -2.8730881, -0.8967180, -1.6165857, 1.6249700
2: -4.1984229, -2.3881874, -4.2174678, -2.3592517, -1.6519039, 1.6425204
3: -12.6284370, -9.9407396, -12.6534681, -9.9015198, -2.1674099, 2.1605368
4: -6.0398083, -4.3091712, -6.0742283, -4.2827578, -1.5423131, 1.5510526
5: -2.8217437, -1.0503111, -2.8988366, -1.0070162, -1.5819523, 1.5649672
6: 2.2715242, 3.7956080, 2.2078104, 3.8329446, -1.5614204, 1.5877976
7: -10.2367878, -8.1965809, -10.2800446, -8.1260977, -1.8351240, 1.8596611
8: -1.9051766, 0.7185178, -1.9243505, 0.7302704, -2.0963144, 2.1056657
9: -8.4931936, -6.9887705, -8.5070066, -6.9832015, -1.4454610, 1.4517190

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.7947138
time: 4.15 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947109, upper bound: 0.7947139
time: 4.16 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8119059, -3.0521817, -4.7981524, -3.0748608, -1.3482549, 1.3562970
1: -2.8730881, -0.8967180, -2.8572679, -0.9035015, -1.6249704, 1.6165855
2: -4.2174678, -2.3592517, -4.1984229, -2.3881874, -1.6425204, 1.6519040
3: -12.6534681, -9.9015198, -12.6284370, -9.9407396, -2.1605365, 2.1674101
4: -6.0742283, -4.2827578, -6.0398083, -4.3091712, -1.5510526, 1.5423133
5: -2.8988366, -1.0070162, -2.8217437, -1.0503111, -1.5649672, 1.5819523
6: 2.2078104, 3.8329446, 2.2715242, 3.7956080, -1.5877976, 1.5614204
7: -10.2800446, -8.1260977, -10.2367878, -8.1965809, -1.8596611, 1.8351240
8: -1.9243505, 0.7302704, -1.9051766, 0.7185178, -2.1056657, 2.0963147
9: -8.5070066, -6.9832015, -8.4931936, -6.9887705, -1.4517188, 1.4454608

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8161543
time: 3.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947109, upper bound: 0.8161537
time: 3.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8120556, -3.0521393, -4.8120556, -3.0521393, -1.3648925, 1.3648925
1: -2.8732462, -0.8965058, -2.8732462, -0.8965058, -1.6384940, 1.6384945
2: -4.2175093, -2.3583987, -4.2175093, -2.3583987, -1.6526833, 1.6526835
3: -12.6534863, -9.9009495, -12.6534863, -9.9009495, -2.1713543, 2.1713543
4: -6.0748878, -4.2827406, -6.0748878, -4.2827406, -1.5578091, 1.5578091
5: -2.9003108, -1.0070014, -2.9003108, -1.0070014, -1.6021743, 1.6021743
6: 2.2075305, 3.8330717, 2.2075305, 3.8330717, -1.6255412, 1.6255412
7: -10.2802534, -8.1249027, -10.2802534, -8.1249027, -1.8692317, 1.8692315
8: -1.9250176, 0.7302978, -1.9250176, 0.7302978, -2.1181002, 2.1181002
9: -8.5070438, -6.9828806, -8.5070438, -6.9828806, -1.4720907, 1.4720905

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8169987
time: 4.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947134, upper bound: 0.8170003
time: 4.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.01 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7914982, upper bound: 0.7947142
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7947109, upper bound: 0.7947137
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7914982, upper bound: 0.7947138
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7947109, upper bound: 0.7947139
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8161543
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7947109, upper bound: 0.8161537
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8169987
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.01
Output dim: 6, lower bound: -0.7947134, upper bound: 0.8170003

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.7981524, -3.0748608, -1.3326557, 1.3262134
1: -2.8501735, -0.9040058, -2.8572679, -0.9035015, -1.6027956, 1.6088643
2: -4.1936932, -2.3900690, -4.1984229, -2.3881874, -1.6182103, 1.6194367
3: -12.6238947, -9.9421434, -12.6284370, -9.9407396, -2.1302421, 2.1324630
4: -6.0342388, -4.3145370, -6.0398083, -4.3091712, -1.5060401, 1.5077243
5: -2.8191915, -1.0619493, -2.8217437, -1.0503111, -1.5355563, 1.5268223
6: 2.2763240, 3.7948284, 2.2715242, 3.7956080, -1.5192840, 1.5233042
7: -10.2257423, -8.1984425, -10.2367878, -8.1965809, -1.8066874, 1.8174450
8: -1.9029455, 0.7092378, -1.9051766, 0.7185178, -2.0821533, 2.0755372
9: -8.4884434, -7.0022831, -8.4931936, -6.9887705, -1.4339924, 1.4236033

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7914971
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7947142
time: 4.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8530836, -3.0726860, -4.7981510, -3.0748725, -1.3546436, 1.3425574
1: -2.8639083, -0.8718609, -2.8572509, -0.9035026, -1.6180737, 1.6233890
2: -4.2019863, -2.3669374, -4.1984158, -2.3881903, -1.6333694, 1.6419272
3: -12.6302471, -9.9066544, -12.6284332, -9.9407406, -2.1447978, 2.1654947
4: -6.0451002, -4.2876120, -6.0397978, -4.3091764, -1.5170856, 1.5747719
5: -2.9089003, -1.0488088, -2.8217409, -1.0503213, -1.5572581, 1.5455298
6: 2.2675314, 3.8122163, 2.2715335, 3.7956076, -1.5280762, 1.5406828
7: -10.2420273, -8.1229925, -10.2367706, -8.1965818, -1.8272886, 1.8323867
8: -1.9650483, 0.7205434, -1.9051728, 0.7185006, -2.1297717, 2.0900197
9: -8.5841188, -6.9858837, -8.4931870, -6.9888024, -1.4541492, 1.4453423

Time for backsubstitution: 13.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7914971
time: 4.07 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7947142
time: 4.18 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8117838, -3.0522156, -1.3545346, 1.3399749
1: -2.8501735, -0.9040058, -2.8729813, -0.8968930, -1.6095848, 1.6239799
2: -4.1936932, -2.3900690, -4.2174335, -2.3599432, -1.6461902, 1.6388472
3: -12.6238947, -9.9421434, -12.6534576, -9.9019928, -2.1626484, 2.1581368
4: -6.0342388, -4.3145370, -6.0736814, -4.2827692, -1.5319371, 1.5418227
5: -2.8191915, -1.0619493, -2.8976190, -1.0070295, -1.5788608, 1.5526223
6: 2.2763240, 3.7948284, 2.2080414, 3.8328571, -1.5565331, 1.5867870
7: -10.2257423, -8.1984425, -10.2799082, -8.1270905, -1.8230987, 1.8591628
8: -1.9029455, 0.7092378, -1.9238322, 0.7302470, -2.0936675, 2.0958452
9: -8.4884434, -7.0022831, -8.5069771, -6.9833012, -1.4412169, 1.4371102

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7914965
time: 4.50 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7947135
time: 4.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8543205, -3.0725923, -4.8120518, -3.0521536, -1.3651085, 1.3566449
1: -2.8643179, -0.8706729, -2.8732288, -0.8965068, -1.6252913, 1.6379355
2: -4.2023964, -2.3649149, -4.2175045, -2.3584015, -1.6639493, 1.6640515
3: -12.6303911, -9.9052658, -12.6534834, -9.9009495, -2.1741555, 2.1861506
4: -6.0453463, -4.2861686, -6.0748758, -4.2827506, -1.5435789, 1.5918744
5: -2.9124928, -1.0487728, -2.9003086, -1.0070164, -1.5890119, 1.5665317
6: 2.2674251, 3.8127658, 2.2075405, 3.8330712, -1.5656462, 1.6052253
7: -10.2426014, -8.1200628, -10.2802343, -8.1249037, -1.8393967, 1.8648260
8: -1.9684837, 0.7206140, -1.9250109, 0.7302830, -2.1409082, 2.1116991
9: -8.5855732, -6.9854202, -8.5070362, -6.9829187, -1.4624987, 1.4591498

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161534, upper bound: 0.7914966
time: 4.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161534, upper bound: 0.7947137
time: 4.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.8103390, -3.0604093, -4.7981524, -3.0748608, -1.3464777, 1.3481479
1: -2.8659072, -0.8976109, -2.8572679, -0.9035015, -1.6179180, 1.6155218
2: -4.2122245, -2.3604386, -4.1984229, -2.3881874, -1.6371636, 1.6487625
3: -12.6487703, -9.9023247, -12.6284370, -9.9407396, -2.1557419, 2.1652372
4: -6.0694137, -4.2891703, -6.0398083, -4.3091712, -1.5417285, 1.5328453
5: -2.8976047, -1.0186930, -2.8217437, -1.0503111, -1.5625050, 1.5700998
6: 2.2122087, 3.8320680, 2.2715242, 3.7956080, -1.5833993, 1.5605438
7: -10.2689981, -8.1271305, -10.2367878, -8.1965809, -1.8484640, 1.8341775
8: -1.9210391, 0.7209513, -1.9051766, 0.7185178, -2.1019258, 2.0870233
9: -8.5016956, -6.9965315, -8.4931936, -6.9887705, -1.4467926, 1.4308307

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.8129288
time: 4.19 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914960, upper bound: 0.8161538
time: 4.27 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8644028, -3.0499449, -4.7981510, -3.0748725, -1.3651702, 1.3644979
1: -2.8787906, -0.8673958, -2.8572509, -0.9035026, -1.6324084, 1.6286891
2: -4.2202578, -2.3409860, -4.1984158, -2.3881903, -1.6508732, 1.6599447
3: -12.6549988, -9.8695803, -12.6284332, -9.9407406, -2.1687002, 2.1704299
4: -6.0792766, -4.2673349, -6.0397978, -4.3091764, -1.5515771, 1.5944695
5: -2.9801745, -1.0056906, -2.8217409, -1.0503213, -1.5677645, 1.5855951
6: 2.2035723, 3.8483105, 2.2715335, 3.7956076, -1.5920353, 1.5767770
7: -10.2844257, -8.0572243, -10.2367706, -8.1965818, -1.8622324, 1.8368099
8: -1.9774642, 0.7321558, -1.9051728, 0.7185006, -2.1469793, 2.1014719
9: -8.5924807, -6.9809403, -8.4931870, -6.9888024, -1.4649534, 1.4523342

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947131, upper bound: 0.8129281
time: 4.10 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.8161543
time: 4.20 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8120556, -3.0521393, -1.3630574, 1.3567381
1: -2.8660669, -0.8974495, -2.8732462, -0.8965058, -1.6314363, 1.6379218
2: -4.2122607, -2.3597307, -4.2175093, -2.3583987, -1.6473203, 1.6494517
3: -12.6487885, -9.9018717, -12.6534863, -9.9009495, -2.1669450, 2.1689034
4: -6.0700092, -4.2891531, -6.0748878, -4.2827406, -1.5484002, 1.5483055
5: -2.8987956, -1.0186815, -2.9003108, -1.0070014, -1.5994954, 1.5903168
6: 2.2119584, 3.8321991, 2.2075305, 3.8330717, -1.6211133, 1.6246686
7: -10.2692127, -8.1261492, -10.2802534, -8.1249027, -1.8580308, 1.8685353
8: -1.9216106, 0.7209752, -1.9250176, 0.7302978, -2.1142173, 2.1088066
9: -8.5017262, -6.9961996, -8.5070438, -6.9828806, -1.4687204, 1.4574678

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8137748
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8170001
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8664532, -3.0497434, -4.8120518, -3.0521536, -1.3780699, 1.3732414
1: -2.8795907, -0.8652791, -2.8732288, -0.8965068, -1.6464653, 1.6504841
2: -4.2213240, -2.3364532, -4.2175045, -2.3584015, -1.6624010, 1.6722461
3: -12.6553612, -9.8662863, -12.6534834, -9.9009495, -2.1817384, 2.1971111
4: -6.0813241, -4.2634163, -6.0748758, -4.2827506, -1.5599554, 1.6147015
5: -2.9885051, -1.0056159, -2.9003086, -1.0070164, -1.6065345, 1.6090655
6: 2.2027016, 3.8495636, 2.2075405, 3.8330712, -1.6303697, 1.6420231
7: -10.2856026, -8.0505514, -10.2802343, -8.1249037, -1.8787265, 1.8738420
8: -1.9838231, 0.7323143, -1.9250109, 0.7302830, -2.1634035, 2.1233647
9: -8.5964184, -6.9795990, -8.5070362, -6.9829187, -1.4839201, 1.4793618

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8137745
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8169998
time: 4.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.38 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7914971
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7947142
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7914971
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7947142
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7914965
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7947135
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.8161534, upper bound: 0.7914966
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.8161534, upper bound: 0.7947137
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7914985, upper bound: 0.8129288
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7914960, upper bound: 0.8161538
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7947131, upper bound: 0.8129281
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7947155, upper bound: 0.8161543
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8137748
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8170001
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8137745
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8169998

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.7967277, -3.0831375, -1.3244829, 1.3244829
1: -2.8501735, -0.9040058, -2.8501735, -0.9040058, -1.6019039, 1.6019039
2: -4.1936932, -2.3900690, -4.1936932, -2.3900690, -1.6145759, 1.6145754
3: -12.6238947, -9.9421434, -12.6238947, -9.9421434, -2.1279647, 2.1279647
4: -6.0342388, -4.3145370, -6.0342388, -4.3145370, -1.4974895, 1.4974892
5: -2.8191915, -1.0619493, -2.8191915, -1.0619493, -1.5237470, 1.5237470
6: 2.2763240, 3.7948284, 2.2763240, 3.7948284, -1.5185044, 1.5185044
7: -10.2257423, -8.1984425, -10.2257423, -8.1984425, -1.8062520, 1.8062522
8: -1.9029455, 0.7092378, -1.9029455, 0.7092378, -2.0729108, 2.0729105
9: -8.4884434, -7.0022831, -8.4884434, -7.0022831, -1.4194074, 1.4194078

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914950
time: 4.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914907
time: 4.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8525968, -3.0727220, -1.3349805, 1.3463805
1: -2.8501735, -0.9040058, -2.8637671, -0.8723648, -1.6162846, 1.6150014
2: -4.1936932, -2.3900690, -4.2017422, -2.3676314, -1.6361670, 1.6236161
3: -12.6238947, -9.9421434, -12.6301661, -9.9071465, -2.1607294, 2.1351969
4: -6.0342388, -4.3145370, -6.0450029, -4.2885504, -1.5222030, 1.5084350
5: -2.8191915, -1.0619493, -2.9076047, -1.0488205, -1.5363865, 1.5451820
6: 2.2763240, 3.7948284, 2.2675683, 3.8119607, -1.5356367, 1.5272601
7: -10.2257423, -8.1984425, -10.2418175, -8.1240778, -1.8205490, 1.8203180
8: -1.9029455, 0.7092378, -1.9636025, 0.7205157, -2.0840669, 2.1196072
9: -8.4884434, -7.0022831, -8.5831852, -6.9860430, -1.4357371, 1.4388354

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947119
time: 4.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947075
time: 4.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8525968, -3.0727220, -4.7967277, -3.0831375, -1.3463802, 1.3349805
1: -2.8637671, -0.8723648, -2.8501735, -0.9040058, -1.6150017, 1.6162845
2: -4.2017422, -2.3676314, -4.1936932, -2.3900690, -1.6236157, 1.6361673
3: -12.6301661, -9.9071465, -12.6238947, -9.9421434, -2.1351969, 2.1607296
4: -6.0450029, -4.2885504, -6.0342388, -4.3145370, -1.5084348, 1.5222032
5: -2.9076047, -1.0488205, -2.8191915, -1.0619493, -1.5451820, 1.5363863
6: 2.2675683, 3.8119607, 2.2763240, 3.7948284, -1.5272601, 1.5356367
7: -10.2418175, -8.1240778, -10.2257423, -8.1984425, -1.8203182, 1.8205492
8: -1.9636025, 0.7205157, -1.9029455, 0.7092378, -2.1196070, 2.0840671
9: -8.5831852, -6.9860430, -8.4884434, -7.0022831, -1.4388356, 1.4357374

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914943
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914899
time: 4.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8543630, -3.0725901, -4.8543630, -3.0725901, -1.3574047, 1.3574049
1: -2.8643343, -0.8706379, -2.8643343, -0.8706379, -1.6303918, 1.6303918
2: -4.2024002, -2.3648329, -4.2024002, -2.3648329, -1.6396298, 1.6396301
3: -12.6303902, -9.9052105, -12.6303902, -9.9052105, -2.1505351, 2.1505351
4: -6.0453529, -4.2861671, -6.0453529, -4.2861671, -1.5764508, 1.5764503
5: -2.9126296, -1.0487709, -2.9126296, -1.0487709, -1.5588470, 1.5588470
6: 2.2674198, 3.8127804, 2.2674198, 3.8127804, -1.5453606, 1.5453606
7: -10.2426243, -8.1199551, -10.2426243, -8.1199551, -1.8373768, 1.8373768
8: -1.9685886, 0.7206161, -1.9685886, 0.7206161, -2.1319337, 2.1319337
9: -8.5855761, -6.9854002, -8.5855761, -6.9854002, -1.4574227, 1.4574227

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947089, upper bound: 0.7927172
time: 4.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947065, upper bound: 0.7927122
time: 5.24 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8102632, -3.0604405, -1.3463883, 1.3382449
1: -2.8501735, -0.9040058, -2.8657720, -0.8977487, -1.6085494, 1.6169158
2: -4.1936932, -2.3900690, -4.2121944, -2.3610396, -1.6430879, 1.6334953
3: -12.6238947, -9.9421434, -12.6487560, -9.9027081, -2.1604981, 2.1533689
4: -6.0342388, -4.3145370, -6.0689092, -4.2891846, -1.5224948, 1.5325503
5: -2.8191915, -1.0619493, -2.8965950, -1.0187044, -1.5670104, 1.5503290
6: 2.2763240, 3.7948284, 2.2124212, 3.8319557, -1.5556316, 1.5824072
7: -10.2257423, -8.1984425, -10.2688179, -8.1279631, -1.8223002, 1.8479505
8: -1.9029455, 0.7092378, -1.9205551, 0.7209315, -2.0843801, 2.0921800
9: -8.4884434, -7.0022831, -8.5016708, -6.9968123, -1.4265532, 1.4321868

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7914942
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7914900
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8637657, -3.0500064, -1.3568885, 1.3568389
1: -2.8501735, -0.9040058, -2.8785808, -0.8680569, -1.6215069, 1.6292696
2: -4.1936932, -2.3900690, -4.2199202, -2.3423502, -1.6545110, 1.6424184
3: -12.6238947, -9.9421434, -12.6548910, -9.8705969, -2.1654930, 2.1604271
4: -6.0342388, -4.3145370, -6.0786428, -4.2685752, -1.5409274, 1.5422828
5: -2.8191915, -1.0619493, -2.9776039, -1.0057137, -1.5795527, 1.5552716
6: 2.2763240, 3.7948284, 2.2038412, 3.8479524, -1.5716283, 1.5909872
7: -10.2257423, -8.1984425, -10.2841139, -8.0592899, -1.8245373, 1.8616526
8: -1.9029455, 0.7092378, -1.9755471, 0.7321053, -2.0954986, 2.1364319
9: -8.4884434, -7.0022831, -8.5912294, -6.9811697, -1.4427004, 1.4495020

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7947114
time: 4.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7947070
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8541927, -3.0726032, -4.8104277, -3.0603719, -1.3568807, 1.3489873
1: -2.8642671, -0.8707823, -2.8660669, -0.8974495, -1.6220882, 1.6309043
2: -4.2023869, -2.3651648, -4.2122607, -2.3597307, -1.6546385, 1.6583741
3: -12.6303844, -9.9054327, -12.6487885, -9.9018717, -2.1687770, 2.1812730
4: -6.0453191, -4.2861753, -6.0700092, -4.2891531, -1.5340555, 1.5621190
5: -2.9120708, -1.0487776, -2.8987956, -1.0186815, -1.5771010, 1.5638518
6: 2.2674375, 3.8127232, 2.2119584, 3.8321991, -1.5647616, 1.6007648
7: -10.2425365, -8.1203947, -10.2692127, -8.1261492, -1.8382239, 1.8534589
8: -1.9681616, 0.7206063, -1.9216106, 0.7209752, -2.1314306, 2.1045117
9: -8.5855618, -6.9854765, -8.5017262, -6.9961996, -1.4479012, 1.4488735

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161464, upper bound: 0.7914943
time: 4.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161464, upper bound: 0.7914899
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8543630, -3.0725901, -4.8669672, -3.0496898, -1.3675506, 1.3683605
1: -2.8643343, -0.8706379, -2.8798280, -0.8647871, -1.6363003, 1.6442230
2: -4.2024002, -2.3648329, -4.2215014, -2.3351412, -1.6699405, 1.6580961
3: -12.6303902, -9.9052105, -12.6554279, -9.8653612, -2.1763811, 2.1755714
4: -6.0453529, -4.2861671, -6.0818939, -4.2628274, -1.5985870, 1.5927765
5: -2.9126296, -1.0487709, -2.9908116, -1.0055947, -1.5899136, 1.5714757
6: 2.2674198, 3.8127804, 2.2024515, 3.8498480, -1.5824282, 1.6103289
7: -10.2426243, -8.1199551, -10.2859316, -8.0487576, -1.8444686, 1.8680689
8: -1.9685886, 0.7206161, -1.9853408, 0.7323537, -2.1430721, 2.1497989
9: -8.5855761, -6.9854002, -8.5970201, -6.9791985, -1.4647002, 1.4702013

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161469, upper bound: 0.7927167
time: 4.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161469, upper bound: 0.7927123
time: 4.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.8102632, -3.0604405, -4.7967277, -3.0831375, -1.3382452, 1.3463883
1: -2.8657720, -0.8977487, -2.8501735, -0.9040058, -1.6169157, 1.6085496
2: -4.2121944, -2.3610396, -4.1936932, -2.3900690, -1.6334949, 1.6430879
3: -12.6487560, -9.9027081, -12.6238947, -9.9421434, -2.1533692, 2.1604986
4: -6.0689092, -4.2891846, -6.0342388, -4.3145370, -1.5325503, 1.5224953
5: -2.8965950, -1.0187044, -2.8191915, -1.0619493, -1.5503290, 1.5670102
6: 2.2124212, 3.8319557, 2.2763240, 3.7948284, -1.5824072, 1.5556316
7: -10.2688179, -8.1279631, -10.2257423, -8.1984425, -1.8479505, 1.8223002
8: -1.9205551, 0.7209315, -1.9029455, 0.7092378, -2.0921803, 2.0843799
9: -8.5016708, -6.9968123, -8.4884434, -7.0022831, -1.4321871, 1.4265530

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129266
time: 4.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129222
time: 4.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8541927, -3.0726032, -1.3489873, 1.3568807
1: -2.8660669, -0.8974495, -2.8642671, -0.8707823, -1.6309042, 1.6220882
2: -4.2122607, -2.3597307, -4.2023869, -2.3651648, -1.6583743, 1.6546385
3: -12.6487885, -9.9018717, -12.6303844, -9.9054327, -2.1812730, 2.1687770
4: -6.0700092, -4.2891531, -6.0453191, -4.2861753, -1.5621192, 1.5340557
5: -2.8987956, -1.0186815, -2.9120708, -1.0487776, -1.5638514, 1.5771010
6: 2.2119584, 3.8321991, 2.2674375, 3.8127232, -1.6007648, 1.5647616
7: -10.2692127, -8.1261492, -10.2425365, -8.1203947, -1.8534586, 1.8382242
8: -1.9216106, 0.7209752, -1.9681616, 0.7206063, -2.1045117, 2.1314309
9: -8.5017262, -6.9961996, -8.5855618, -6.9854765, -1.4488735, 1.4479015

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161522
time: 4.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161478
time: 4.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8637657, -3.0500064, -4.7967277, -3.0831375, -1.3568392, 1.3568885
1: -2.8785808, -0.8680569, -2.8501735, -0.9040058, -1.6292696, 1.6215069
2: -4.2199202, -2.3423502, -4.1936932, -2.3900690, -1.6424184, 1.6545106
3: -12.6548910, -9.8705969, -12.6238947, -9.9421434, -2.1604269, 2.1654928
4: -6.0786428, -4.2685752, -6.0342388, -4.3145370, -1.5422831, 1.5409274
5: -2.9776039, -1.0057137, -2.8191915, -1.0619493, -1.5552719, 1.5795527
6: 2.2038412, 3.8479524, 2.2763240, 3.7948284, -1.5909872, 1.5716283
7: -10.2841139, -8.0592899, -10.2257423, -8.1984425, -1.8616529, 1.8245373
8: -1.9755471, 0.7321053, -1.9029455, 0.7092378, -2.1364317, 2.0954986
9: -8.5912294, -6.9811697, -8.4884434, -7.0022831, -1.4495020, 1.4427001

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947060, upper bound: 0.8129260
time: 4.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947084, upper bound: 0.8129215
time: 4.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8669672, -3.0496898, -4.8543630, -3.0725901, -1.3683605, 1.3675504
1: -2.8798280, -0.8647871, -2.8643343, -0.8706379, -1.6442230, 1.6363003
2: -4.2215014, -2.3351412, -4.2024002, -2.3648329, -1.6580958, 1.6699402
3: -12.6554279, -9.8653612, -12.6303902, -9.9052105, -2.1755714, 2.1763809
4: -6.0818939, -4.2628274, -6.0453529, -4.2861671, -1.5927768, 1.5985866
5: -2.9908116, -1.0055947, -2.9126296, -1.0487709, -1.5714760, 1.5899138
6: 2.2024515, 3.8498480, 2.2674198, 3.8127804, -1.6103289, 1.5824282
7: -10.2859316, -8.0487576, -10.2426243, -8.1199551, -1.8680689, 1.8444684
8: -1.9853408, 0.7323537, -1.9685886, 0.7206161, -2.1497989, 2.1430721
9: -8.5970201, -6.9791985, -8.5855761, -6.9854002, -1.4702010, 1.4647005

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135207
time: 4.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135165
time: 4.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8104277, -3.0603719, -1.3549030, 1.3549030
1: -2.8660669, -0.8974495, -2.8660669, -0.8974495, -1.6308632, 1.6308637
2: -4.2122607, -2.3597307, -4.2122607, -2.3597307, -1.6440883, 1.6440880
3: -12.6487885, -9.9018717, -12.6487885, -9.9018717, -2.1644933, 2.1644936
4: -6.0700092, -4.2891531, -6.0700092, -4.2891531, -1.5388961, 1.5388966
5: -2.8987956, -1.0186815, -2.8987956, -1.0186815, -1.5876384, 1.5876381
6: 2.2119584, 3.8321991, 2.2119584, 3.8321991, -1.6202407, 1.6202407
7: -10.2692127, -8.1261492, -10.2692127, -8.1261492, -1.8573360, 1.8573363
8: -1.9216106, 0.7209752, -1.9216106, 0.7209752, -2.1049237, 2.1049240
9: -8.5017262, -6.9961996, -8.5017262, -6.9961996, -1.4540973, 1.4540975

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137729
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137685
time: 4.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8661208, -3.0497751, -1.3655612, 1.3697968
1: -2.8660669, -0.8974495, -2.8794792, -0.8656535, -1.6432509, 1.6437387
2: -4.2122607, -2.3597307, -4.2210836, -2.3370624, -1.6661212, 1.6539621
3: -12.6487885, -9.9018717, -12.6552811, -9.8667431, -2.1922421, 2.1723640
4: -6.0700092, -4.2891531, -6.0810366, -4.2643619, -1.5629582, 1.5501249
5: -2.8987956, -1.0186815, -2.9873214, -1.0056245, -1.6003036, 1.5943356
6: 2.2119584, 3.8321991, 2.2028177, 3.8493290, -1.6373706, 1.6293814
7: -10.2692127, -8.1261492, -10.2854271, -8.0515366, -1.8620980, 1.8714592
8: -1.9216106, 0.7209752, -1.9827099, 0.7322876, -2.1161613, 2.1533844
9: -8.5017262, -6.9961996, -8.5954781, -6.9797850, -1.4706032, 1.4685843

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169982
time: 4.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169940
time: 4.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8661208, -3.0497751, -4.8104277, -3.0603719, -1.3697968, 1.3655612
1: -2.8794792, -0.8656535, -2.8660669, -0.8974495, -1.6437387, 1.6432509
2: -4.2210836, -2.3370624, -4.2122607, -2.3597307, -1.6539621, 1.6661211
3: -12.6552811, -9.8667431, -12.6487885, -9.9018717, -2.1723640, 2.1922424
4: -6.0810366, -4.2643619, -6.0700092, -4.2891531, -1.5501246, 1.5629585
5: -2.9873214, -1.0056245, -2.8987956, -1.0186815, -1.5943356, 1.6003036
6: 2.2028177, 3.8493290, 2.2119584, 3.8321991, -1.6293814, 1.6373706
7: -10.2854271, -8.0515366, -10.2692127, -8.1261492, -1.8714595, 1.8620980
8: -1.9827099, 0.7322876, -1.9216106, 0.7209752, -2.1533842, 2.1161613
9: -8.5954781, -6.9797850, -8.5017262, -6.9961996, -1.4685845, 1.4706032

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137728
time: 5.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137682
time: 4.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8671603, -3.0496688, -4.8671603, -3.0496688, -1.3806839, 1.3806841
1: -2.8799314, -0.8646218, -2.8799314, -0.8646218, -1.6569967, 1.6569970
2: -4.2215219, -2.3345556, -4.2215219, -2.3345556, -1.6699762, 1.6699765
3: -12.6554384, -9.8649588, -12.6554384, -9.8649588, -2.1870213, 2.1870213
4: -6.0821409, -4.2628155, -6.0821409, -4.2628155, -1.6163168, 1.6163168
5: -2.9917998, -1.0055847, -2.9917998, -1.0055847, -1.6081975, 1.6081977
6: 2.2023408, 3.8499389, 2.2023408, 3.8499389, -1.6475980, 1.6475980
7: -10.2860727, -8.0480108, -10.2860727, -8.0480108, -1.8783133, 1.8783133
8: -1.9858670, 0.7323687, -1.9858670, 0.7323687, -2.1621556, 2.1621554
9: -8.5970411, -6.9790187, -8.5970411, -6.9790187, -1.4865291, 1.4865294

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142789
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142745
time: 4.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.32 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914950
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914907
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947119
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947075
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914943
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914899
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947089, upper bound: 0.7927172
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947065, upper bound: 0.7927122
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7914942
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7914900
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7947114
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7947070
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8161464, upper bound: 0.7914943
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8161464, upper bound: 0.7914899
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8161469, upper bound: 0.7927167
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.8161469, upper bound: 0.7927123
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129266
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129222
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161522
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161478
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947060, upper bound: 0.8129260
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947084, upper bound: 0.8129215
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135207
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135165
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137729
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137685
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169982
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169940
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137728
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137682
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142789
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.32
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142745

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.7963915, -3.0831957, -4.7967277, -3.0831375, -1.3238661, 1.3244267
1: -2.8500171, -0.9047991, -2.8501735, -0.9040058, -1.6015921, 1.6011183
2: -4.1935730, -2.3902187, -4.1936932, -2.3900690, -1.6144404, 1.6143363
3: -12.6236572, -9.9425879, -12.6238947, -9.9421434, -2.1277497, 2.1275206
4: -6.0341082, -4.3147926, -6.0342388, -4.3145370, -1.4973869, 1.4969664
5: -2.8191175, -1.0622549, -2.8191915, -1.0619493, -1.5236764, 1.5231972
6: 2.2765188, 3.7947831, 2.2763240, 3.7948284, -1.5183096, 1.5184591
7: -10.2256594, -8.1992712, -10.2257423, -8.1984425, -1.8061852, 1.8054326
8: -1.9025915, 0.7090173, -1.9029455, 0.7092378, -2.0725641, 2.0724850
9: -8.4883442, -7.0029688, -8.4884434, -7.0022831, -1.4193578, 1.4184971

Time for backsubstitution: 12.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914919
time: 4.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
time: 5.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8136315, -3.0713849, -4.7967253, -3.0831375, -1.3524799, 1.3369784
1: -2.8852556, -0.9020772, -2.8501730, -0.9040116, -1.6251342, 1.6058662
2: -4.1985049, -2.3839588, -4.1936922, -2.3900700, -1.6216269, 1.6235104
3: -12.6263542, -9.9161177, -12.6238918, -9.9421492, -2.1315157, 2.1596532
4: -6.0494347, -4.2856245, -6.0342379, -4.3145385, -1.5144711, 1.5297225
5: -2.8298650, -1.0522091, -2.8191905, -1.0619521, -1.5342579, 1.5480533
6: 2.2654133, 3.7969077, 2.2763271, 3.7948279, -1.5294147, 1.5205805
7: -10.2423840, -8.1975193, -10.2257423, -8.1984463, -1.8204985, 1.8093476
8: -1.9221399, 0.7117667, -1.9029434, 0.7092364, -2.0994439, 2.0844481
9: -8.5095425, -6.9970574, -8.4884415, -7.0022840, -1.4375033, 1.4284554

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914921
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.7963915, -3.0831957, -4.8525968, -3.0727220, -1.3343635, 1.3463256
1: -2.8500171, -0.9047991, -2.8637671, -0.8723648, -1.6159751, 1.6142159
2: -4.1935730, -2.3902187, -4.2017422, -2.3676314, -1.6360321, 1.6233767
3: -12.6236572, -9.9425879, -12.6301661, -9.9071465, -2.1605144, 2.1347525
4: -6.0341082, -4.3147926, -6.0450029, -4.2885504, -1.5221004, 1.5079119
5: -2.8191175, -1.0622549, -2.9076047, -1.0488205, -1.5363159, 1.5446227
6: 2.2765188, 3.7947831, 2.2675683, 3.8119607, -1.5354419, 1.5272148
7: -10.2256594, -8.1992712, -10.2418175, -8.1240778, -1.8204775, 1.8194985
8: -1.9025915, 0.7090173, -1.9636025, 0.7205157, -2.0837207, 2.1191611
9: -8.4883442, -7.0029688, -8.5831852, -6.9860430, -1.4356875, 1.4379230

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947070
time: 4.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947071
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8136315, -3.0713849, -4.8525949, -3.0727234, -1.3629770, 1.3519733
1: -2.8852556, -0.9020772, -2.8637652, -0.8723712, -1.6271861, 1.6189632
2: -4.1985049, -2.3839588, -4.2017403, -2.3676338, -1.6432180, 1.6325502
3: -12.6263542, -9.9161177, -12.6301632, -9.9071522, -2.1643267, 2.1668854
4: -6.0494347, -4.2856245, -6.0450010, -4.2885513, -1.5391855, 1.5406680
5: -2.8298650, -1.0522091, -2.9076037, -1.0488226, -1.5468965, 1.5575562
6: 2.2654133, 3.7969077, 2.2675714, 3.8119597, -1.5465465, 1.5293362
7: -10.2423840, -8.1975193, -10.2418156, -8.1240797, -1.8236976, 1.8234134
8: -1.9221399, 0.7117667, -1.9635992, 0.7205131, -2.1105986, 2.1244717
9: -8.5095425, -6.9970574, -8.5831833, -6.9860458, -1.4539330, 1.4417195

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947071
time: 4.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947070
time: 4.41 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.04 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914919
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914921
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947070
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947071
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947071
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947070
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914943
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914899
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947089, upper bound: 0.7927172
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947065, upper bound: 0.7927122
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7914942
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7914900
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7947114
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8129258, upper bound: 0.7947070
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8161464, upper bound: 0.7914943
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8161464, upper bound: 0.7914899
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8161469, upper bound: 0.7927167
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.8161469, upper bound: 0.7927123
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129266
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129222
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161522
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161478
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947060, upper bound: 0.8129260
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947084, upper bound: 0.8129215
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135207
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135165
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137729
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137685
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169982
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169940
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137728
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137682
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142789
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142745
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.568244457244873
rel_dist={6: [-0.8170414609587353, 0.8170398270121066]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039341, upper bound: 0.5937624
time: 4.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6051967, upper bound: 0.6051922
time: 4.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.83
Output dim: 6, lower bound: -0.6039341, upper bound: 0.5937624
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.83
Output dim: 6, lower bound: -0.6051967, upper bound: 0.6051922

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8039060, -3.0711708, -1.1855927, 1.1883311
1: -2.8572679, -0.9035015, -2.8615298, -0.9012308, -1.4420223, 1.4448862
2: -4.1984229, -2.3881874, -4.2059588, -2.3855734, -1.4901175, 1.4949081
3: -12.6284370, -9.9407396, -12.6386900, -9.9395714, -1.9107602, 1.9185276
4: -6.0398083, -4.3091712, -6.0434775, -4.2985716, -1.3967621, 1.3912508
5: -2.8217437, -1.0503111, -2.8247194, -1.0313241, -1.4245360, 1.4082818
6: 2.2715242, 3.7956080, 2.2675519, 3.8117661, -1.4982324, 1.4853868
7: -10.2367878, -8.1965809, -10.2553816, -8.1947374, -1.6303875, 1.6480765
8: -1.9051766, 0.7185178, -1.9100540, 0.7231512, -1.9280214, 1.9289601
9: -8.4931936, -6.9887705, -8.4977942, -6.9872389, -1.3394854, 1.3401070

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937635
time: 4.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937659
time: 5.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8120537, -3.0521393, -4.8109589, -3.0668550, -1.2015896, 1.2174602
1: -2.8732445, -0.8965081, -2.8668013, -0.8982893, -1.4608169, 1.4591410
2: -4.2175097, -2.3584073, -4.2153416, -2.3826575, -1.5070591, 1.5340068
3: -12.6534863, -9.9009542, -12.6510620, -9.9383640, -1.9339094, 1.9588258
4: -6.0748825, -4.2827406, -6.0474720, -4.2851405, -1.4445317, 1.4157677
5: -2.9002964, -1.0070014, -2.8278701, -1.0085013, -1.4690483, 1.4399812
6: 2.2075326, 3.8330705, 2.2630315, 3.8312418, -1.5489206, 1.5139606
7: -10.2802534, -8.1249142, -10.2777634, -8.1927633, -1.6544495, 1.6865525
8: -1.9250109, 0.7302978, -1.9165046, 0.7287591, -1.9550943, 1.9477940
9: -8.5070438, -6.9828863, -8.5035810, -6.9852877, -1.3640385, 1.3541906

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937659, upper bound: 0.6039316
time: 4.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937659, upper bound: 0.6051967
time: 4.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.11 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937635
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937659
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 6, lower bound: -0.5937659, upper bound: 0.6039316
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 6, lower bound: -0.5937659, upper bound: 0.6051967

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.7981524, -3.0748608, -1.1824462, 1.1824462
1: -2.8572679, -0.9035015, -2.8572679, -0.9035015, -1.4390445, 1.4390445
2: -4.1984229, -2.3881874, -4.1984229, -2.3881874, -1.4873414, 1.4873414
3: -12.6284370, -9.9407396, -12.6284370, -9.9407396, -1.9073246, 1.9073246
4: -6.0398083, -4.3091712, -6.0398083, -4.3091712, -1.3869624, 1.3869624
5: -2.8217437, -1.0503111, -2.8217437, -1.0503111, -1.4053187, 1.4053187
6: 2.2715242, 3.7956080, 2.2715242, 3.7956080, -1.4818234, 1.4818234
7: -10.2367878, -8.1965809, -10.2367878, -8.1965809, -1.6295567, 1.6295564
8: -1.9051766, 0.7185178, -1.9051766, 0.7185178, -1.9234471, 1.9234471
9: -8.4931936, -6.9887705, -8.4931936, -6.9887705, -1.3343277, 1.3343279

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904141, upper bound: 0.5937455
time: 4.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937478, upper bound: 0.5937454
time: 4.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8118277, -3.0522027, -1.2043371, 1.1962457
1: -2.8572679, -0.9035015, -2.8730195, -0.8968298, -1.4458485, 1.4541955
2: -4.1984229, -2.3881874, -4.2174439, -2.3596957, -1.5156274, 1.5067650
3: -12.6284370, -9.9407396, -12.6534615, -9.9018230, -1.9336200, 1.9330418
4: -6.0398083, -4.3091712, -6.0738773, -4.2827654, -1.4129097, 1.4213037
5: -2.8217437, -1.0503111, -2.8980565, -1.0070257, -1.4318740, 1.4257103
6: 2.2715242, 3.7956080, 2.2079601, 3.8328891, -1.5141537, 1.5124083
7: -10.2367878, -8.1965809, -10.2799568, -8.1267347, -1.6442747, 1.6558111
8: -1.9051766, 0.7185178, -1.9240160, 0.7302547, -1.9349689, 1.9439616
9: -8.4931936, -6.9887705, -8.5069866, -6.9832649, -1.3415697, 1.3478422

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904141, upper bound: 0.5937456
time: 5.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937478, upper bound: 0.5937483
time: 4.18 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8118277, -3.0522027, -4.7981524, -3.0748608, -1.1962459, 1.2043369
1: -2.8730195, -0.8968298, -2.8572679, -0.9035015, -1.4541955, 1.4458480
2: -4.2174439, -2.3596957, -4.1984229, -2.3881874, -1.5067649, 1.5156275
3: -12.6534615, -9.9018230, -12.6284370, -9.9407396, -1.9330418, 1.9336197
4: -6.0738773, -4.2827654, -6.0398083, -4.3091712, -1.4213042, 1.4129095
5: -2.8980565, -1.0070257, -2.8217437, -1.0503111, -1.4257102, 1.4318738
6: 2.2079601, 3.8328891, 2.2715242, 3.7956080, -1.5124080, 1.5141537
7: -10.2799568, -8.1267347, -10.2367878, -8.1965809, -1.6558108, 1.6442752
8: -1.9240160, 0.7302547, -1.9051766, 0.7185178, -1.9439616, 1.9349692
9: -8.5069866, -6.9832649, -8.4931936, -6.9887705, -1.3478422, 1.3415694

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904141, upper bound: 0.6039149
time: 4.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937478, upper bound: 0.6039147
time: 4.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8120556, -3.0521393, -4.8120556, -3.0521393, -1.2120497, 1.2120497
1: -2.8732462, -0.8965058, -2.8732462, -0.8965058, -1.4661603, 1.4661603
2: -4.2175093, -2.3583987, -4.2175093, -2.3583987, -1.5136619, 1.5136619
3: -12.6534863, -9.9009495, -12.6534863, -9.9009495, -1.9360723, 1.9360723
4: -6.0748878, -4.2827406, -6.0748878, -4.2827406, -1.4251182, 1.4251180
5: -2.9003108, -1.0070014, -2.9003108, -1.0070014, -1.4554529, 1.4554532
6: 2.2075305, 3.8330717, 2.2075305, 3.8330717, -1.5324252, 1.5324252
7: -10.2802534, -8.1249027, -10.2802534, -8.1249027, -1.6690242, 1.6690242
8: -1.9250176, 0.7302978, -1.9250176, 0.7302978, -1.9566975, 1.9566977
9: -8.5070438, -6.9828806, -8.5070438, -6.9828806, -1.3661003, 1.3661005

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904141, upper bound: 0.6051773
time: 5.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937478, upper bound: 0.6039189
time: 5.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5904141, upper bound: 0.5937455
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5937478, upper bound: 0.5937454
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5904141, upper bound: 0.5937456
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5937478, upper bound: 0.5937483
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5904141, upper bound: 0.6039149
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5937478, upper bound: 0.6039147
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5904141, upper bound: 0.6051773
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.90
Output dim: 6, lower bound: -0.5937478, upper bound: 0.6039189

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.7981524, -3.0748608, -1.1807158, 1.1742737
1: -2.8501735, -0.9040058, -2.8572679, -0.9035015, -1.4320841, 1.4381528
2: -4.1936932, -2.3900690, -4.1984229, -2.3881874, -1.4824805, 1.4837071
3: -12.6238947, -9.9421434, -12.6284370, -9.9407396, -1.9028261, 1.9050469
4: -6.0342388, -4.3145370, -6.0398083, -4.3091712, -1.3767271, 1.3784118
5: -2.8191915, -1.0619493, -2.8217437, -1.0503111, -1.4022431, 1.3935091
6: 2.2763240, 3.7948284, 2.2715242, 3.7956080, -1.4764256, 1.4788346
7: -10.2257423, -8.1984425, -10.2367878, -8.1965809, -1.6183634, 1.6291208
8: -1.9029455, 0.7092378, -1.9051766, 0.7185178, -1.9208202, 1.9142046
9: -8.4884434, -7.0022831, -8.4931936, -6.9887705, -1.3301325, 1.3197432

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5904108
time: 4.43 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5937456
time: 4.62 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8524861, -3.0727229, -4.7981501, -3.0748796, -1.1999183, 1.1853721
1: -2.8638165, -0.8726310, -2.8572373, -0.9035025, -1.4454277, 1.4491946
2: -4.2013464, -2.3673844, -4.1984124, -2.3881917, -1.4924829, 1.5055895
3: -12.6300430, -9.9070425, -12.6284304, -9.9407406, -1.9122419, 1.9316316
4: -6.0449781, -4.2902718, -6.0397892, -4.3091798, -1.3876429, 1.4241549
5: -2.9077806, -1.0488164, -2.8217397, -1.0503249, -1.4179714, 1.4067676
6: 2.2675560, 3.8117490, 2.2715416, 3.7956061, -1.4921191, 1.4962955
7: -10.2418346, -8.1240625, -10.2367601, -8.1965828, -1.6331096, 1.6412809
8: -1.9628735, 0.7205062, -1.9051707, 0.7184894, -1.9608026, 1.9256883
9: -8.5814991, -6.9859867, -8.4931850, -6.9888229, -1.3452344, 1.3366289

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5904109
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5937456
time: 4.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8113523, -3.0522523, -1.2025599, 1.1875992
1: -2.8501735, -0.9040058, -2.8728540, -0.8971285, -1.4385729, 1.4531596
2: -4.1936932, -2.3900690, -4.2174025, -2.3605614, -1.5097575, 1.5030912
3: -12.6238947, -9.9421434, -12.6534424, -9.9024763, -1.9282105, 1.9304969
4: -6.0342388, -4.3145370, -6.0729752, -4.2827849, -1.4024248, 1.4116516
5: -2.8191915, -1.0619493, -2.8965611, -1.0070426, -1.4287677, 1.4122810
6: 2.2763240, 3.7948284, 2.2083516, 3.8327761, -1.5086541, 1.5090275
7: -10.2257423, -8.1984425, -10.2797794, -8.1277618, -1.6320479, 1.6552970
8: -1.9029455, 0.7092378, -1.9231553, 0.7302363, -1.9323263, 1.9335074
9: -8.4884434, -7.0022831, -8.5069609, -6.9836779, -1.3367627, 1.3332355

Time for backsubstitution: 12.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5904098
time: 5.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5937447
time: 5.03 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8540006, -3.0726142, -4.8120508, -3.0521631, -1.2104442, 1.1994722
1: -2.8642433, -0.8710400, -2.8732169, -0.8965083, -1.4526598, 1.4627503
2: -4.2021599, -2.3652840, -4.2174988, -2.3584032, -1.5237596, 1.5208087
3: -12.6303148, -9.9055433, -12.6534786, -9.9009523, -1.9380338, 1.9454644
4: -6.0452814, -4.2871208, -6.0748634, -4.2827559, -1.4141891, 1.4423838
5: -2.9117403, -1.0487795, -2.9003072, -1.0070276, -1.4358947, 1.4276042
6: 2.2674437, 3.8125644, 2.2075481, 3.8330698, -1.5199986, 1.5154982
7: -10.2424774, -8.1207237, -10.2802191, -8.1249056, -1.6490235, 1.6605935
8: -1.9674389, 0.7205958, -1.9250095, 0.7302704, -1.9697475, 1.9473803
9: -8.5846310, -6.9855022, -8.5070324, -6.9829407, -1.3546188, 1.3504605

Time for backsubstitution: 13.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039148, upper bound: 0.5904100
time: 4.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039148, upper bound: 0.5937483
time: 4.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.8102908, -3.0604286, -4.7981524, -3.0748608, -1.1944993, 1.1961899
1: -2.8658204, -0.8976988, -2.8572679, -0.9035015, -1.4471354, 1.4448028
2: -4.2122040, -2.3608241, -4.1984229, -2.3881874, -1.5014110, 1.5125126
3: -12.6487617, -9.9025698, -12.6284370, -9.9407396, -1.9282649, 1.9314620
4: -6.0690913, -4.2891803, -6.0398083, -4.3091712, -1.4120140, 1.4034591
5: -2.8969593, -1.0187006, -2.8217437, -1.0503111, -1.4233572, 1.4200402
6: 2.2123456, 3.8319960, 2.2715242, 3.7956080, -1.5071294, 1.5110559
7: -10.2688828, -8.1276636, -10.2367878, -8.1965809, -1.6446075, 1.6434255
8: -1.9207284, 0.7209404, -1.9051766, 0.7185178, -1.9402714, 1.9256794
9: -8.5016785, -6.9967117, -8.4931936, -6.9887705, -1.3429193, 1.3269179

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6005928
time: 4.41 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6039147
time: 4.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8636122, -3.0500102, -4.7981501, -3.0748796, -1.2068615, 1.2072816
1: -2.8786521, -0.8684112, -2.8572373, -0.9035025, -1.4597161, 1.4530852
2: -4.2193975, -2.3418810, -4.1984124, -2.3881917, -1.5103848, 1.5195959
3: -12.6547337, -9.8703499, -12.6284304, -9.9407406, -1.9359450, 1.9364185
4: -6.0787740, -4.2706404, -6.0397892, -4.3091798, -1.4216905, 1.4381096
5: -2.9780521, -1.0057082, -2.8217397, -1.0503249, -1.4281101, 1.4326611
6: 2.2037659, 3.8476748, 2.2715416, 3.7956061, -1.5173514, 1.5153015
7: -10.2841387, -8.0591869, -10.2367601, -8.1965828, -1.6583033, 1.6452479
8: -1.9745858, 0.7320957, -1.9051707, 0.7184894, -1.9767022, 1.9371204
9: -8.5890121, -6.9810963, -8.4931850, -6.9888229, -1.3546376, 1.3435938

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6005930
time: 4.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6039140
time: 4.37 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8120556, -3.0521393, -1.2102146, 1.2038951
1: -2.8660669, -0.8974495, -2.8732462, -0.8965058, -1.4591022, 1.4655876
2: -4.2122607, -2.3597307, -4.2175093, -2.3583987, -1.5082984, 1.5104301
3: -12.6487885, -9.9018717, -12.6534863, -9.9009495, -1.9316630, 1.9336214
4: -6.0700092, -4.2891531, -6.0748878, -4.2827406, -1.4157088, 1.4156144
5: -2.8987956, -1.0186815, -2.9003108, -1.0070014, -1.4527745, 1.4435958
6: 2.2119584, 3.8321991, 2.2075305, 3.8330717, -1.5273876, 1.5292900
7: -10.2692127, -8.1261492, -10.2802534, -8.1249027, -1.6578252, 1.6678796
8: -1.9216106, 0.7209752, -1.9250176, 0.7302978, -1.9528151, 1.9474041
9: -8.5017262, -6.9961996, -8.5070438, -6.9828806, -1.3627300, 1.3514776

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925962, upper bound: 0.6018574
time: 4.90 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925983, upper bound: 0.6051779
time: 7.03 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8657060, -3.0498028, -4.8120508, -3.0521631, -1.2224019, 1.2151327
1: -2.8794260, -0.8662313, -2.8732169, -0.8965083, -1.4721370, 1.4744846
2: -4.2205291, -2.3373466, -4.2174988, -2.3584032, -1.5181935, 1.5320630
3: -12.6551123, -9.8670330, -12.6534786, -9.9009523, -1.9409842, 1.9552598
4: -6.0808401, -4.2666621, -6.0748634, -4.2827559, -1.4267080, 1.4602947
5: -2.9864578, -1.0056307, -2.9003072, -1.0070276, -1.4591103, 1.4568741
6: 2.2028737, 3.8489413, 2.2075481, 3.8330698, -1.5426445, 1.5392776
7: -10.2852869, -8.0524158, -10.2802191, -8.1249056, -1.6725450, 1.6720135
8: -1.9810665, 0.7322578, -1.9250095, 0.7302704, -1.9939666, 1.9589441
9: -8.5932064, -6.9798717, -8.5070324, -6.9829407, -1.3724339, 1.3684735

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5959072, upper bound: 0.6018564
time: 4.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5959072, upper bound: 0.6051758
time: 4.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.49 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5904108
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5937456
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5904109
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5937456
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5904098
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5937447
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.6039148, upper bound: 0.5904100
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.6039148, upper bound: 0.5937483
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6005928
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6039147
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6005930
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6039140
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5925962, upper bound: 0.6018574
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5925983, upper bound: 0.6051779
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5959072, upper bound: 0.6018564
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.49
Output dim: 6, lower bound: -0.5959072, upper bound: 0.6051758

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.7967277, -3.0831375, -1.1725435, 1.1725433
1: -2.8501735, -0.9040058, -2.8501735, -0.9040058, -1.4311924, 1.4311924
2: -4.1936932, -2.3900690, -4.1936932, -2.3900690, -1.4788461, 1.4788457
3: -12.6238947, -9.9421434, -12.6238947, -9.9421434, -1.9005487, 1.9005487
4: -6.0342388, -4.3145370, -6.0342388, -4.3145370, -1.3681765, 1.3681765
5: -2.8191915, -1.0619493, -2.8191915, -1.0619493, -1.3904338, 1.3904338
6: 2.2763240, 3.7948284, 2.2763240, 3.7948284, -1.4734366, 1.4734366
7: -10.2257423, -8.1984425, -10.2257423, -8.1984425, -1.6179280, 1.6179280
8: -1.9029455, 0.7092378, -1.9029455, 0.7092378, -1.9115777, 1.9115777
9: -8.4884434, -7.0022831, -8.4884434, -7.0022831, -1.3155475, 1.3155477

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5901522
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5904034
time: 4.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8518982, -3.0727661, -1.1829991, 1.1916325
1: -2.8501735, -0.9040058, -2.8636487, -0.8732444, -1.4420500, 1.4441891
2: -4.1936932, -2.3900690, -4.2010407, -2.3682089, -1.4996617, 1.4871792
3: -12.6238947, -9.9421434, -12.6299438, -9.9076290, -1.9268522, 1.9073877
4: -6.0342388, -4.3145370, -6.0448608, -4.2914515, -1.3894448, 1.3789747
5: -2.8191915, -1.0619493, -2.9062331, -1.0488305, -1.4030530, 1.4058404
6: 2.2763240, 3.7948284, 2.2676013, 3.8114369, -1.4905505, 1.4828880
7: -10.2257423, -8.1984425, -10.2415829, -8.1253643, -1.6293101, 1.6318727
8: -1.9029455, 0.7092378, -1.9611173, 0.7204721, -1.9226980, 1.9504390
9: -8.4884434, -7.0022831, -8.5803270, -6.9861770, -1.3317845, 1.3298686

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5934871
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5937384
time: 4.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8518982, -3.0727661, -4.7967277, -3.0831375, -1.1916323, 1.1829991
1: -2.8636487, -0.8732444, -2.8501735, -0.9040058, -1.4441891, 1.4420501
2: -4.2010407, -2.3682089, -4.1936932, -2.3900690, -1.4871793, 1.4996614
3: -12.6299438, -9.9076290, -12.6238947, -9.9421434, -1.9073880, 1.9268522
4: -6.0448608, -4.2914515, -6.0342388, -4.3145370, -1.3789749, 1.3894453
5: -2.9062331, -1.0488305, -2.8191915, -1.0619493, -1.4058404, 1.4030530
6: 2.2676013, 3.8114369, 2.2763240, 3.7948284, -1.4828880, 1.4905503
7: -10.2415829, -8.1253643, -10.2257423, -8.1984425, -1.6318727, 1.6293099
8: -1.9611173, 0.7204721, -1.9029455, 0.7092378, -1.9504390, 1.9226980
9: -8.5803270, -6.9861770, -8.4884434, -7.0022831, -1.3298688, 1.3317847

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937403, upper bound: 0.5901515
time: 4.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937403, upper bound: 0.5904035
time: 4.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8543630, -3.0725901, -4.8543630, -3.0725901, -1.2028160, 1.2028162
1: -2.8643343, -0.8706379, -2.8643343, -0.8706379, -1.4566054, 1.4566056
2: -4.2024002, -2.3648329, -4.2024002, -2.3648329, -1.4998708, 1.4998704
3: -12.6303902, -9.9052105, -12.6303902, -9.9052105, -1.9185972, 1.9185972
4: -6.0453529, -4.2861671, -6.0453529, -4.2861671, -1.4324079, 1.4324082
5: -2.9126296, -1.0487709, -2.9126296, -1.0487709, -1.4199321, 1.4199321
6: 2.2674198, 3.8127804, 2.2674198, 3.8127804, -1.4982584, 1.4982584
7: -10.2426243, -8.1199551, -10.2426243, -8.1199551, -1.6470635, 1.6470635
8: -1.9685886, 0.7206161, -1.9685886, 0.7206161, -1.9665532, 1.9665532
9: -8.5855761, -6.9854002, -8.5855761, -6.9854002, -1.3504405, 1.3504405

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5906702
time: 4.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5909193
time: 4.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8097782, -3.0604722, -1.1944187, 1.1858084
1: -2.8501735, -0.9040058, -2.8656938, -0.8979839, -1.4374855, 1.4461288
2: -4.1936932, -2.3900690, -4.2121673, -2.3615611, -1.5068481, 1.4977469
3: -12.6238947, -9.9421434, -12.6487455, -9.9031563, -1.9260237, 1.9257205
4: -6.0342388, -4.3145370, -6.0682759, -4.2891946, -1.3929873, 1.4024742
5: -2.8191915, -1.0619493, -2.8956695, -1.0187154, -1.4169374, 1.4098446
6: 2.2763240, 3.7948284, 2.2127023, 3.8319187, -1.5055950, 1.5037353
7: -10.2257423, -8.1984425, -10.2687569, -8.1284904, -1.6313350, 1.6441197
8: -1.9029455, 0.7092378, -1.9199510, 0.7209253, -1.9230404, 1.9298615
9: -8.4884434, -7.0022831, -8.5016603, -6.9970407, -1.3220592, 1.3283181

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005899, upper bound: 0.5901514
time: 4.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005876, upper bound: 0.5904021
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.8629565, -3.0500700, -1.2048860, 1.1981826
1: -2.8501735, -0.9040058, -2.8785014, -0.8690202, -1.4456010, 1.4584854
2: -4.1936932, -2.3900690, -4.2190361, -2.3427031, -1.5137961, 1.5059277
3: -12.6238947, -9.9421434, -12.6546059, -9.8710327, -1.9309652, 1.9323511
4: -6.0342388, -4.3145370, -6.0778227, -4.2719455, -1.4072523, 1.4120128
5: -2.8191915, -1.0619493, -2.9763896, -1.0057216, -1.4295554, 1.4145240
6: 2.2763240, 3.7948284, 2.2041571, 3.8473420, -1.5096390, 1.5130079
7: -10.2257423, -8.1984425, -10.2839241, -8.0604973, -1.6327236, 1.6577711
8: -1.9029455, 0.7092378, -1.9728456, 0.7320623, -1.9341288, 1.9655962
9: -8.4884434, -7.0022831, -8.5875006, -6.9814987, -1.3382368, 1.3389683

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005899, upper bound: 0.5934869
time: 4.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005876, upper bound: 0.5937371
time: 5.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8537149, -3.0726335, -4.8104277, -3.0603719, -1.2021837, 1.1970189
1: -2.8641775, -0.8713702, -2.8660669, -0.8974495, -1.4513018, 1.4556122
2: -4.2019405, -2.3655994, -4.2122607, -2.3597307, -1.5184000, 1.5154290
3: -12.6302433, -9.9057837, -12.6487885, -9.9018717, -1.9348917, 1.9405904
4: -6.0452237, -4.2880116, -6.0700092, -4.2891531, -1.4046409, 1.4270430
5: -2.9110880, -1.0487859, -2.8987956, -1.0186815, -1.4238918, 1.4249231
6: 2.2674613, 3.8123815, 2.2119584, 3.8321991, -1.5157051, 1.5099912
7: -10.2423716, -8.1213007, -10.2692127, -8.1261492, -1.6478305, 1.6490257
8: -1.9665003, 0.7205787, -1.9216106, 0.7209752, -1.9598522, 1.9431551
9: -8.5837498, -6.9855762, -8.5017262, -6.9961996, -1.3394232, 1.3449464

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039068, upper bound: 0.5901510
time: 4.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039068, upper bound: 0.5904021
time: 4.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8543630, -3.0725901, -4.8668709, -3.0497024, -1.2129526, 1.2104197
1: -2.8643343, -0.8706379, -2.8797755, -0.8648679, -1.4613895, 1.4692457
2: -4.2024002, -2.3648329, -4.2214904, -2.3354282, -1.5299187, 1.5183238
3: -12.6303902, -9.9052105, -12.6554203, -9.8655586, -1.9426665, 1.9435673
4: -6.0453529, -4.2861671, -6.0817728, -4.2628336, -1.4497616, 1.4486626
5: -2.9126296, -1.0487709, -2.9903262, -1.0056000, -1.4370160, 1.4324473
6: 2.2674198, 3.8127804, 2.2025070, 3.8498046, -1.5244532, 1.5229359
7: -10.2426243, -8.1199551, -10.2858648, -8.0491238, -1.6539834, 1.6642597
8: -1.9685886, 0.7206161, -1.9850826, 0.7323458, -1.9725890, 1.9851811
9: -8.5855761, -6.9854002, -8.5970078, -6.9792871, -1.3576214, 1.3624585

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039073, upper bound: 0.5906696
time: 4.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039073, upper bound: 0.5909181
time: 5.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.8097782, -3.0604722, -4.7967277, -3.0831375, -1.1858087, 1.1944187
1: -2.8656938, -0.8979839, -2.8501735, -0.9040058, -1.4461288, 1.4374855
2: -4.2121673, -2.3615611, -4.1936932, -2.3900690, -1.4977469, 1.5068480
3: -12.6487455, -9.9031563, -12.6238947, -9.9421434, -1.9257205, 1.9260240
4: -6.0682759, -4.2891946, -6.0342388, -4.3145370, -1.4024744, 1.3929875
5: -2.8956695, -1.0187154, -2.8191915, -1.0619493, -1.4098446, 1.4169374
6: 2.2127023, 3.8319187, 2.2763240, 3.7948284, -1.5037355, 1.5055950
7: -10.2687569, -8.1284904, -10.2257423, -8.1984425, -1.6441197, 1.6313350
8: -1.9199510, 0.7209253, -1.9029455, 0.7092378, -1.9298615, 1.9230409
9: -8.5016603, -6.9970407, -8.4884434, -7.0022831, -1.3283181, 1.3220594

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6003370
time: 4.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6005852
time: 4.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8537149, -3.0726335, -1.1970189, 1.2021832
1: -2.8660669, -0.8974495, -2.8641775, -0.8713702, -1.4556122, 1.4513018
2: -4.2122607, -2.3597307, -4.2019405, -2.3655994, -1.5154290, 1.5184001
3: -12.6487885, -9.9018717, -12.6302433, -9.9057837, -1.9405909, 1.9348917
4: -6.0700092, -4.2891531, -6.0452237, -4.2880116, -1.4270430, 1.4046412
5: -2.8987956, -1.0186815, -2.9110880, -1.0487859, -1.4249229, 1.4238918
6: 2.2119584, 3.8321991, 2.2674613, 3.8123815, -1.5099912, 1.5157049
7: -10.2692127, -8.1261492, -10.2423716, -8.1213007, -1.6490257, 1.6478307
8: -1.9216106, 0.7209752, -1.9665003, 0.7205787, -1.9431553, 1.9598522
9: -8.5017262, -6.9961996, -8.5837498, -6.9855762, -1.3449466, 1.3394232

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6036588
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6039070
time: 4.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8629565, -3.0500700, -4.7967277, -3.0831375, -1.1981826, 1.2048862
1: -2.8785014, -0.8690202, -2.8501735, -0.9040058, -1.4584856, 1.4456010
2: -4.2190361, -2.3427031, -4.1936932, -2.3900690, -1.5059276, 1.5137961
3: -12.6546059, -9.8710327, -12.6238947, -9.9421434, -1.9323509, 1.9309652
4: -6.0778227, -4.2719455, -6.0342388, -4.3145370, -1.4120131, 1.4072523
5: -2.9763896, -1.0057216, -2.8191915, -1.0619493, -1.4145241, 1.4295554
6: 2.2041571, 3.8473420, 2.2763240, 3.7948284, -1.5130081, 1.5096388
7: -10.2839241, -8.0604973, -10.2257423, -8.1984425, -1.6577711, 1.6327233
8: -1.9728456, 0.7320623, -1.9029455, 0.7092378, -1.9655962, 1.9341288
9: -8.5875006, -6.9814987, -8.4884434, -7.0022831, -1.3389683, 1.3382370

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937403, upper bound: 0.6003365
time: 4.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937403, upper bound: 0.6005853
time: 4.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8668709, -3.0497024, -4.8543630, -3.0725901, -1.2104197, 1.2129524
1: -2.8797755, -0.8648679, -2.8643343, -0.8706379, -1.4692457, 1.4613895
2: -4.2214904, -2.3354282, -4.2024002, -2.3648329, -1.5183234, 1.5299182
3: -12.6554203, -9.8655586, -12.6303902, -9.9052105, -1.9435673, 1.9426663
4: -6.0817728, -4.2628336, -6.0453529, -4.2861671, -1.4486628, 1.4497616
5: -2.9903262, -1.0056000, -2.9126296, -1.0487709, -1.4324474, 1.4370161
6: 2.2025070, 3.8498046, 2.2674198, 3.8127804, -1.5229361, 1.5244534
7: -10.2858648, -8.0491238, -10.2426243, -8.1199551, -1.6642597, 1.6539831
8: -1.9850826, 0.7323458, -1.9685886, 0.7206161, -1.9851813, 1.9725893
9: -8.5970078, -6.9792871, -8.5855761, -6.9854002, -1.3624587, 1.3576214

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937408, upper bound: 0.6006252
time: 5.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937408, upper bound: 0.6008734
time: 4.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8104277, -3.0603719, -1.2020600, 1.2020600
1: -2.8660669, -0.8974495, -2.8660669, -0.8974495, -1.4585295, 1.4585295
2: -4.2122607, -2.3597307, -4.2122607, -2.3597307, -1.5050664, 1.5050664
3: -12.6487885, -9.9018717, -12.6487885, -9.9018717, -1.9292114, 1.9292116
4: -6.0700092, -4.2891531, -6.0700092, -4.2891531, -1.4062057, 1.4062054
5: -2.8987956, -1.0186815, -2.8987956, -1.0186815, -1.4409170, 1.4409171
6: 2.2119584, 3.8321991, 2.2119584, 3.8321991, -1.5242522, 1.5242522
7: -10.2692127, -8.1261492, -10.2692127, -8.1261492, -1.6566806, 1.6566806
8: -1.9216106, 0.7209752, -1.9216106, 0.7209752, -1.9435215, 1.9435215
9: -8.5017262, -6.9961996, -8.5017262, -6.9961996, -1.3481073, 1.3481073

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6016034
time: 4.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6018497
time: 4.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8652987, -3.0498402, -1.2126567, 1.2141118
1: -2.8660669, -0.8974495, -2.8792934, -0.8666933, -1.4672062, 1.4712462
2: -4.2122607, -2.3597307, -4.2202263, -2.3380840, -1.5257809, 1.5142174
3: -12.6487885, -9.9018717, -12.6550112, -9.8675880, -1.9503994, 1.9364100
4: -6.0700092, -4.2891531, -6.0804911, -4.2678480, -1.4261408, 1.4168117
5: -2.8987956, -1.0186815, -2.9850197, -1.0056427, -1.4535530, 1.4468369
6: 2.2119584, 3.8321991, 2.2030134, 3.8486538, -1.5336738, 1.5339921
7: -10.2692127, -8.1261492, -10.2850733, -8.0536165, -1.6601458, 1.6706324
8: -1.9216106, 0.7209752, -1.9796953, 0.7322264, -1.9547062, 1.9837787
9: -8.5017262, -6.9961996, -8.5920286, -6.9800954, -1.3644745, 1.3569248

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6049232
time: 5.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6051693
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8652987, -3.0498402, -4.8104277, -3.0603719, -1.2141118, 1.2126567
1: -2.8792934, -0.8666933, -2.8660669, -0.8974495, -1.4712458, 1.4672062
2: -4.2202263, -2.3380840, -4.2122607, -2.3597307, -1.5142174, 1.5257810
3: -12.6550112, -9.8675880, -12.6487885, -9.9018717, -1.9364102, 1.9503996
4: -6.0804911, -4.2678480, -6.0700092, -4.2891531, -1.4168115, 1.4261408
5: -2.9850197, -1.0056427, -2.8987956, -1.0186815, -1.4468369, 1.4535533
6: 2.2030134, 3.8486538, 2.2119584, 3.8321991, -1.5339921, 1.5336740
7: -10.2850733, -8.0536165, -10.2692127, -8.1261492, -1.6706324, 1.6601460
8: -1.9796953, 0.7322264, -1.9216106, 0.7209752, -1.9837790, 1.9547062
9: -8.5920286, -6.9800954, -8.5017262, -6.9961996, -1.3569245, 1.3644748

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5958998, upper bound: 0.6016027
time: 4.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5958998, upper bound: 0.6018497
time: 4.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8671603, -3.0496688, -4.8671603, -3.0496688, -1.2251883, 1.2251880
1: -2.8799314, -0.8646218, -2.8799314, -0.8646218, -1.4815814, 1.4815814
2: -4.2215219, -2.3345556, -4.2215219, -2.3345556, -1.5269241, 1.5269241
3: -12.6554384, -9.8649588, -12.6554384, -9.8649588, -1.9472160, 1.9472160
4: -6.0821409, -4.2628155, -6.0821409, -4.2628155, -1.4688878, 1.4688877
5: -2.9917998, -1.0055847, -2.9917998, -1.0055847, -1.4614766, 1.4614767
6: 2.2023408, 3.8499389, 2.2023408, 3.8499389, -1.5475564, 1.5475566
7: -10.2860727, -8.0480108, -10.2860727, -8.0480108, -1.6776578, 1.6776578
8: -1.9858670, 0.7323687, -1.9858670, 0.7323687, -1.9977880, 1.9977880
9: -8.5970411, -6.9790187, -8.5970411, -6.9790187, -1.3774083, 1.3774085

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5959003, upper bound: 0.6018370
time: 4.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5959003, upper bound: 0.6020706
time: 4.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.61 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5901522
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5904034
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5934871
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5937384
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937403, upper bound: 0.5901515
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937403, upper bound: 0.5904035
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5906702
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5909193
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6005899, upper bound: 0.5901514
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6005876, upper bound: 0.5904021
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6005899, upper bound: 0.5934869
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6005876, upper bound: 0.5937371
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6039068, upper bound: 0.5901510
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6039068, upper bound: 0.5904021
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6039073, upper bound: 0.5906696
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.6039073, upper bound: 0.5909181
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6003370
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6005852
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6036588
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5904066, upper bound: 0.6039070
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937403, upper bound: 0.6003365
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937403, upper bound: 0.6005853
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937408, upper bound: 0.6006252
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5937408, upper bound: 0.6008734
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6016034
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6018497
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6049232
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5925894, upper bound: 0.6051693
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5958998, upper bound: 0.6016027
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5958998, upper bound: 0.6018497
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5959003, upper bound: 0.6018370
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.61
Output dim: 6, lower bound: -0.5959003, upper bound: 0.6020706

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.7963915, -3.0831957, -4.7967277, -3.0831375, -1.1719265, 1.1724870
1: -2.8500171, -0.9047991, -2.8501735, -0.9040058, -1.4308805, 1.4304068
2: -4.1935730, -2.3902187, -4.1936932, -2.3900690, -1.4787107, 1.4786066
3: -12.6236572, -9.9425879, -12.6238947, -9.9421434, -1.9003336, 1.9001045
4: -6.0341082, -4.3147926, -6.0342388, -4.3145370, -1.3680744, 1.3676536
5: -2.8191175, -1.0622549, -2.8191915, -1.0619493, -1.3903632, 1.3898840
6: 2.2765188, 3.7947831, 2.2763240, 3.7948284, -1.4730198, 1.4732933
7: -10.2256594, -8.1992712, -10.2257423, -8.1984425, -1.6178613, 1.6171086
8: -1.9025915, 0.7090173, -1.9029455, 0.7092378, -1.9112315, 1.9111524
9: -8.4883442, -7.0029688, -8.4884434, -7.0022831, -1.3154979, 1.3146369

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901536, upper bound: 0.5901509
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901536, upper bound: 0.5901496
time: 4.49 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8119717, -3.0716639, -4.7967238, -3.0831389, -1.1976576, 1.1847718
1: -2.8834500, -0.9021467, -2.8501732, -0.9040166, -1.4503717, 1.4337417
2: -4.1982145, -2.3841133, -4.1936903, -2.3900714, -1.4854715, 1.4873906
3: -12.6262789, -9.9178085, -12.6238890, -9.9421549, -1.9038284, 1.9300163
4: -6.0491982, -4.2887688, -6.0342355, -4.3145409, -1.3849320, 1.3964119
5: -2.8297834, -1.0534532, -2.8191900, -1.0619533, -1.4008667, 1.4105215
6: 2.2660160, 3.7968991, 2.2763290, 3.7948279, -1.4842176, 1.4757071
7: -10.2423162, -8.1975422, -10.2257404, -8.1984501, -1.6300588, 1.6197412
8: -1.9210150, 0.7117364, -1.9029408, 0.7092345, -1.9361076, 1.9220686
9: -8.5094090, -6.9976163, -8.4884396, -7.0022874, -1.3303690, 1.3225689

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901536, upper bound: 0.5904042
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901536, upper bound: 0.5904034
time: 4.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.7963915, -3.0831957, -4.8518982, -3.0727661, -1.1823823, 1.1915779
1: -2.8500171, -0.9047991, -2.8636487, -0.8732444, -1.4417408, 1.4434035
2: -4.1935730, -2.3902187, -4.2010407, -2.3682089, -1.4995267, 1.4869399
3: -12.6236572, -9.9425879, -12.6299438, -9.9076290, -1.9266372, 1.9069436
4: -6.0341082, -4.3147926, -6.0448608, -4.2914515, -1.3893428, 1.3784516
5: -2.8191175, -1.0622549, -2.9062331, -1.0488305, -1.4029825, 1.4052812
6: 2.2765188, 3.7947831, 2.2676013, 3.8114369, -1.4901335, 1.4827447
7: -10.2256594, -8.1992712, -10.2415829, -8.1253643, -1.6292386, 1.6310532
8: -1.9025915, 0.7090173, -1.9611173, 0.7204721, -1.9223518, 1.9499929
9: -8.4883442, -7.0029688, -8.5803270, -6.9861770, -1.3317349, 1.3289557

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901534, upper bound: 0.5934844
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901534, upper bound: 0.5934831
time: 4.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8119717, -3.0716639, -4.8518939, -3.0727675, -1.2081130, 1.1969664
1: -2.8834500, -0.9021467, -2.8636467, -0.8732548, -1.4519758, 1.4467382
2: -4.1982145, -2.3841133, -4.2010374, -2.3682127, -1.5062873, 1.4957242
3: -12.6262789, -9.9178085, -12.6299400, -9.9076395, -1.9301705, 1.9368563
4: -6.0491982, -4.2887688, -6.0448575, -4.2914553, -1.4062002, 1.4072104
5: -2.8297834, -1.0534532, -2.9062309, -1.0488350, -1.4134865, 1.4164340
6: 2.2660160, 3.7968991, 2.2676058, 3.8114376, -1.4999826, 1.4851594
7: -10.2423162, -8.1975422, -10.2415810, -8.1253719, -1.6323321, 1.6336851
8: -1.9210150, 0.7117364, -1.9611115, 0.7204700, -1.9472265, 1.9550674
9: -8.5094090, -6.9976163, -8.5803223, -6.9861822, -1.3467093, 1.3325484

Time for backsubstitution: 13.02 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.525653600692749
rel_dist={6: [-0.6052081522674859, 0.6052060697814263]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466822, upper bound: 0.4433082
time: 4.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479277, upper bound: 0.4479263
time: 4.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 6, lower bound: -0.4466822, upper bound: 0.4433082
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 6, lower bound: -0.4479277, upper bound: 0.4479263

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8002834, -3.0734711, -1.0823238, 1.0833395
1: -2.8572679, -0.9035015, -2.8588457, -0.9026766, -1.3263645, 1.3274064
2: -4.1984229, -2.3881874, -4.2012253, -2.3871851, -1.3978920, 1.3996675
3: -12.6284370, -9.9407396, -12.6322651, -9.9402819, -1.7569976, 1.7599092
4: -6.0398083, -4.3091712, -6.0412230, -4.3053699, -1.3043778, 1.3024101
5: -2.8217437, -1.0503111, -2.8229032, -1.0432165, -1.3236260, 1.3176126
6: 2.2715242, 3.7956080, 2.2700167, 3.8016393, -1.4141250, 1.4093444
7: -10.2367878, -8.1965809, -10.2437344, -8.1958656, -1.5043213, 1.5109305
8: -1.9051766, 0.7185178, -1.9069371, 0.7202439, -1.8175945, 1.8178859
9: -8.4931936, -6.9887705, -8.4948864, -6.9881954, -1.2670085, 1.2672691

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4433079
time: 4.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4433064
time: 5.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8120146, -3.0521522, -4.8109550, -3.0668576, -1.0996585, 1.1161540
1: -2.8731935, -0.8965639, -2.8667982, -0.8982921, -1.3469658, 1.3441820
2: -4.2174978, -2.3586414, -4.2153339, -2.3826594, -1.4143629, 1.4421737
3: -12.6534824, -9.9011078, -12.6510515, -9.9383659, -1.7769675, 1.8029926
4: -6.0747056, -4.2827468, -6.0474701, -4.2851472, -1.3580999, 1.3272583
5: -2.8999031, -1.0070052, -2.8278680, -1.0085170, -1.3762653, 1.3459065
6: 2.2076073, 3.8330288, 2.2630336, 3.8312311, -1.4737535, 1.4351761
7: -10.2801847, -8.1252317, -10.2777500, -8.1927633, -1.5219884, 1.5594087
8: -1.9248180, 0.7302904, -1.9164996, 0.7287579, -1.8472800, 1.8402288
9: -8.5070343, -6.9829884, -8.5035782, -6.9852886, -1.2924027, 1.2849159

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4466819
time: 4.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4479281
time: 10.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.33 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.33
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4433079
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 28.33
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4433064
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.33
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4466819
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.33
Output dim: 6, lower bound: -0.4433085, upper bound: 0.4479281

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8113384, -3.0522420, -4.7981524, -3.0748608, -1.0945053, 1.1030049
1: -2.8729153, -0.8972008, -2.8572679, -0.9035015, -1.3402996, 1.3315716
2: -4.2171450, -2.3601537, -4.1984229, -2.3881874, -1.4159913, 1.4238851
3: -12.6533575, -9.9022293, -12.6284370, -9.9407396, -1.7800598, 1.7771027
4: -6.0732799, -4.2832680, -6.0398083, -4.3091712, -1.3343825, 1.3257012
5: -2.8971434, -1.0070477, -2.8217437, -1.0503111, -1.3317015, 1.3298869
6: 2.2082324, 3.8327134, 2.2715242, 3.7956080, -1.4369276, 1.4308803
7: -10.2798328, -8.1273661, -10.2367878, -8.1965809, -1.5198808, 1.5166030
8: -1.9229436, 0.7302246, -1.9051766, 0.7185178, -1.8350167, 1.8273859
9: -8.5062809, -6.9836082, -8.4931936, -6.9887705, -1.2777777, 1.2717569

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416932, upper bound: 0.4463302
time: 4.62 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4433032, upper bound: 0.4466764
time: 4.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8120556, -3.0521393, -4.8120556, -3.0521393, -1.1101544, 1.1101544
1: -2.8732462, -0.8965058, -2.8732462, -0.8965058, -1.3512712, 1.3512707
2: -4.2175093, -2.3583987, -4.2175093, -2.3583987, -1.4209809, 1.4209808
3: -12.6534863, -9.9009495, -12.6534863, -9.9009495, -1.7792177, 1.7792177
4: -6.0748878, -4.2827406, -6.0748878, -4.2827406, -1.3366573, 1.3366575
5: -2.9003108, -1.0070014, -2.9003108, -1.0070014, -1.3576391, 1.3576391
6: 2.2075305, 3.8330717, 2.2075305, 3.8330717, -1.4536893, 1.4536893
7: -10.2802534, -8.1249027, -10.2802534, -8.1249027, -1.5352538, 1.5352538
8: -1.9250176, 0.7302978, -1.9250176, 0.7302978, -1.8490958, 1.8490961
9: -8.5070438, -6.9828806, -8.5070438, -6.9828806, -1.2944746, 1.2944748

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416932, upper bound: 0.4479248
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4433031, upper bound: 0.4466759
time: 7.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.86 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 6, lower bound: -0.4416932, upper bound: 0.4463302
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 6, lower bound: -0.4433032, upper bound: 0.4466764
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 6, lower bound: -0.4416932, upper bound: 0.4479248
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.86
Output dim: 6, lower bound: -0.4433031, upper bound: 0.4466759

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.8092356, -3.0604951, -4.7973332, -3.0797892, -1.0873291, 1.0938380
1: -2.8657022, -0.8983817, -2.8530421, -0.9037985, -1.3327036, 1.3259053
2: -4.2116413, -2.3616419, -4.1955795, -2.3892670, -1.4081712, 1.4164932
3: -12.6485844, -9.9033623, -12.6257229, -9.9415379, -1.7736983, 1.7713521
4: -6.0679994, -4.2903347, -6.0364904, -4.3123026, -1.3194714, 1.3094621
5: -2.8952780, -1.0187440, -2.8203011, -1.0572393, -1.3203738, 1.3163090
6: 2.2128496, 3.8317733, 2.2743831, 3.7951503, -1.4294019, 1.4244518
7: -10.2687702, -8.1287231, -10.2302113, -8.1976471, -1.5084310, 1.5081797
8: -1.9189196, 0.7208841, -1.9038587, 0.7129884, -1.8246799, 1.8165388
9: -8.5005159, -6.9970636, -8.4904213, -6.9968147, -1.2636611, 1.2540488

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416930, upper bound: 0.4450636
time: 4.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416930, upper bound: 0.4463300
time: 4.53 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8630795, -3.0500445, -4.7981491, -3.0748870, -1.1006598, 1.1024821
1: -2.8787141, -0.8689227, -2.8572197, -0.9035041, -1.3447180, 1.3354676
2: -4.2187071, -2.3412590, -4.1984062, -2.3881941, -1.4165614, 1.4253759
3: -12.6545029, -9.8701019, -12.6284285, -9.9407425, -1.7794018, 1.7794330
4: -6.0778761, -4.2731886, -6.0397758, -4.3091831, -1.3344073, 1.3330957
5: -2.9786797, -1.0057259, -2.8217375, -1.0503302, -1.3328950, 1.3276916
6: 2.2041354, 3.8473499, 2.2715530, 3.7956052, -1.4389839, 1.4319742
7: -10.2842054, -8.0588131, -10.2367487, -8.1965837, -1.5193005, 1.5171223
8: -1.9732611, 0.7320635, -1.9051695, 0.7184727, -1.8623915, 1.8275533
9: -8.5862055, -6.9813333, -8.4931822, -6.9888439, -1.2804751, 1.2701309

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4429580, upper bound: 0.4450636
time: 4.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4429585, upper bound: 0.4450646
time: 4.40 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8111072, -3.0570421, -1.1034665, 1.1009324
1: -2.8660669, -0.8974495, -2.8689694, -0.8970571, -1.3438785, 1.3464966
2: -4.2122607, -2.3597307, -4.2143517, -2.3591819, -1.4136667, 1.4145417
3: -12.6487885, -9.9018717, -12.6506863, -9.9014883, -1.7733488, 1.7741446
4: -6.0700092, -4.2891531, -6.0719700, -4.2867041, -1.3216462, 1.3215344
5: -2.8987956, -1.0186815, -2.8994284, -1.0139508, -1.3475935, 1.3442316
6: 2.2119584, 3.8321991, 2.2101812, 3.8325593, -1.4467802, 1.4475634
7: -10.2692127, -8.1261492, -10.2736845, -8.1256294, -1.5233889, 1.5271640
8: -1.9216106, 0.7209752, -1.9230163, 0.7247455, -1.8396835, 1.8375258
9: -8.5017262, -6.9961996, -8.5039301, -6.9908161, -1.2810290, 1.2772651

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436943, upper bound: 0.4468524
time: 4.67 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436943, upper bound: 0.4479253
time: 4.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8651094, -3.0498536, -4.8120499, -3.0521722, -1.1185994, 1.1097147
1: -2.8792977, -0.8669947, -2.8731987, -0.8965095, -1.3559012, 1.3570669
2: -4.2198844, -2.3380499, -4.2174945, -2.3584051, -1.4222765, 1.4384680
3: -12.6549110, -9.8676195, -12.6534758, -9.9009533, -1.7803507, 1.7940271
4: -6.0804596, -4.2692881, -6.0748482, -4.2827592, -1.3378072, 1.3570418
5: -2.9848418, -1.0056434, -2.9003053, -1.0070348, -1.3607349, 1.3554070
6: 2.2030182, 3.8484430, 2.2075586, 3.8330688, -1.4597936, 1.4588392
7: -10.2850380, -8.0539045, -10.2802029, -8.1249065, -1.5348072, 1.5373001
8: -1.9788582, 0.7322133, -1.9250052, 0.7302551, -1.8807588, 1.8493228
9: -8.5906105, -6.9800863, -8.5070286, -6.9829683, -1.2977731, 1.2935605

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4447274, upper bound: 0.4468526
time: 4.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4447275, upper bound: 0.4479254
time: 4.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.14 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4416930, upper bound: 0.4450636
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4416930, upper bound: 0.4463300
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4429580, upper bound: 0.4450636
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4429585, upper bound: 0.4450646
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4436943, upper bound: 0.4468524
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4436943, upper bound: 0.4479253
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4447274, upper bound: 0.4468526
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.14
Output dim: 6, lower bound: -0.4447275, upper bound: 0.4479254

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.8085551, -3.0605154, -4.7967277, -3.0831375, -1.0832291, 1.0930817
1: -2.8656790, -0.8987854, -2.8501735, -0.9040058, -1.3322926, 1.3225870
2: -4.2111082, -2.3618572, -4.1936932, -2.3900690, -1.4061265, 1.4142506
3: -12.6483383, -9.9035940, -12.6238947, -9.9421434, -1.7726622, 1.7691207
4: -6.0676708, -4.2927880, -6.0342388, -4.3145370, -1.3155727, 1.3030677
5: -2.8947206, -1.0187516, -2.8191915, -1.0619493, -1.3146996, 1.3149297
6: 2.2129962, 3.8315210, 2.2763240, 3.7948284, -1.4279337, 1.4220197
7: -10.2687597, -8.1292191, -10.2257423, -8.1984425, -1.5082488, 1.5034494
8: -1.9180813, 0.7208774, -1.9029455, 0.7092378, -1.8198695, 1.8154407
9: -8.4983101, -6.9971523, -8.4884434, -7.0022831, -1.2556891, 1.2517962

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416839, upper bound: 0.4448137
time: 4.20 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416840, upper bound: 0.4450606
time: 4.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.8099289, -3.0603960, -4.8532329, -3.0726624, -1.0951817, 1.0989926
1: -2.8660271, -0.8976511, -2.8641057, -0.8719903, -1.3385682, 1.3370619
2: -4.2122355, -2.3600755, -4.2014222, -2.3659563, -1.4187565, 1.4232416
3: -12.6487837, -9.9022150, -12.6300831, -9.9060926, -1.7799234, 1.7757666
4: -6.0695109, -4.2891593, -6.0451255, -4.2901678, -1.3285170, 1.3181646
5: -2.8981628, -1.0187018, -2.9101913, -1.0487912, -1.3238258, 1.3216231
6: 2.2121964, 3.8321977, 2.2674809, 3.8120046, -1.4342129, 1.4308181
7: -10.2692118, -8.1264210, -10.2422190, -8.1221581, -1.5125217, 1.5141568
8: -1.9211361, 0.7209592, -1.9647467, 0.7205474, -1.8347034, 1.8450139
9: -8.5017204, -6.9963446, -8.5816269, -6.9856567, -1.2703056, 1.2659264

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416840, upper bound: 0.4460800
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416838, upper bound: 0.4463274
time: 4.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8622980, -3.0500715, -4.7967277, -3.0831375, -1.0908513, 1.1018252
1: -2.8787415, -0.8692629, -2.8501735, -0.9040058, -1.3448696, 1.3274847
2: -4.2172146, -2.3411129, -4.1936932, -2.3900690, -1.4133449, 1.4194615
3: -12.6538658, -9.8700924, -12.6238947, -9.9421434, -1.7764797, 1.7737935
4: -6.0769644, -4.2772083, -6.0342388, -4.3145370, -1.3247595, 1.3162420
5: -2.9787467, -1.0057464, -2.8191915, -1.0619493, -1.3183432, 1.3206038
6: 2.2045450, 3.8470938, 2.2763240, 3.7948284, -1.4350481, 1.4258852
7: -10.2843161, -8.0588827, -10.2257423, -8.1984425, -1.5158300, 1.5045047
8: -1.9728332, 0.7320476, -1.9029455, 0.7092378, -1.8509278, 1.8265524
9: -8.5831804, -6.9815216, -8.4884434, -7.0022831, -1.2611895, 1.2638860

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416838, upper bound: 0.4448172
time: 4.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4416837, upper bound: 0.4450634
time: 4.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8660388, -3.0497737, -4.8543630, -3.0725901, -1.1011810, 1.1069460
1: -2.8796806, -0.8654437, -2.8643343, -0.8706379, -1.3514984, 1.3428574
2: -4.2210217, -2.3359795, -4.2024002, -2.3648329, -1.4245753, 1.4325223
3: -12.6552734, -9.8661242, -12.6303902, -9.9052105, -1.7860525, 1.7826376
4: -6.0801415, -4.2643566, -6.0453529, -4.2861671, -1.3475714, 1.3462998
5: -2.9891224, -1.0056386, -2.9126296, -1.0487709, -1.3332987, 1.3320258
6: 2.2032204, 3.8495665, 2.2674198, 3.8127804, -1.4447844, 1.4390326
7: -10.2857857, -8.0499535, -10.2426243, -8.1199551, -1.5251915, 1.5219698
8: -1.9835782, 0.7322965, -1.9685886, 0.7206161, -1.8703914, 1.8572533
9: -8.5955343, -6.9797840, -8.5855761, -6.9854002, -1.2867680, 1.2823813

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4429493, upper bound: 0.4448135
time: 6.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4429493, upper bound: 0.4450619
time: 7.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8104277, -3.0603719, -1.1001649, 1.1001649
1: -2.8660669, -0.8974495, -2.8660669, -0.8974495, -1.3436399, 1.3436401
2: -4.2122607, -2.3597307, -4.2122607, -2.3597307, -1.4123855, 1.4123853
3: -12.6487885, -9.9018717, -12.6487885, -9.9018717, -1.7723567, 1.7723570
4: -6.0700092, -4.2891531, -6.0700092, -4.2891531, -1.3177447, 1.3177447
5: -2.8987956, -1.0186815, -2.8987956, -1.0186815, -1.3431029, 1.3431031
6: 2.2119584, 3.8321991, 2.2119584, 3.8321991, -1.4455163, 1.4455163
7: -10.2692127, -8.1261492, -10.2692127, -8.1261492, -1.5229101, 1.5229101
8: -1.9216106, 0.7209752, -1.9216106, 0.7209752, -1.8359199, 1.8359199
9: -8.5017262, -6.9961996, -8.5017262, -6.9961996, -1.2754521, 1.2754521

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4466069
time: 5.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4468509
time: 5.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8646193, -3.0498977, -1.1107056, 1.1102092
1: -2.8660669, -0.8974495, -2.8791561, -0.8675270, -1.3496969, 1.3562400
2: -4.2122607, -2.3597307, -4.2195272, -2.3388300, -1.4321537, 1.4209498
3: -12.6487885, -9.9018717, -12.6547947, -9.8682232, -1.7889924, 1.7789974
4: -6.0700092, -4.2891531, -6.0799599, -4.2705021, -1.3343279, 1.3277376
5: -2.8987956, -1.0186815, -2.9832907, -1.0056562, -1.3487751, 1.3481326
6: 2.2119584, 3.8321991, 2.2032216, 3.8481207, -1.4532142, 1.4549363
7: -10.2692127, -8.1261492, -10.2848129, -8.0551910, -1.5252440, 1.5304692
8: -1.9216106, 0.7209752, -1.9773040, 0.7321796, -1.8470640, 1.8702936
9: -8.5017262, -6.9961996, -8.5892229, -6.9803553, -1.2837105, 1.2821131

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4476797
time: 5.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4479237
time: 4.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8646193, -3.0498977, -4.8104277, -3.0603719, -1.1102095, 1.1107056
1: -2.8791561, -0.8675270, -2.8660669, -0.8974495, -1.3562398, 1.3496970
2: -4.2195272, -2.3388300, -4.2122607, -2.3597307, -1.4209495, 1.4321537
3: -12.6547947, -9.8682232, -12.6487885, -9.9018717, -1.7789972, 1.7889924
4: -6.0799599, -4.2705021, -6.0700092, -4.2891531, -1.3277373, 1.3343282
5: -2.9832907, -1.0056562, -2.8987956, -1.0186815, -1.3481326, 1.3487749
6: 2.2032216, 3.8481207, 2.2119584, 3.8321991, -1.4549363, 1.4532139
7: -10.2848129, -8.0551910, -10.2692127, -8.1261492, -1.5304694, 1.5252442
8: -1.9773040, 0.7321796, -1.9216106, 0.7209752, -1.8702936, 1.8470640
9: -8.5892229, -6.9803553, -8.5017262, -6.9961996, -1.2821131, 1.2837107

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436865, upper bound: 0.4466100
time: 5.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4436864, upper bound: 0.4468514
time: 6.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8671603, -3.0496688, -4.8671603, -3.0496688, -1.1186447, 1.1186445
1: -2.8799314, -0.8646218, -2.8799314, -0.8646218, -1.3636103, 1.3636103
2: -4.2215219, -2.3345556, -4.2215219, -2.3345556, -1.4315557, 1.4315559
3: -12.6554384, -9.8649588, -12.6554384, -9.8649588, -1.7873459, 1.7873461
4: -6.0821409, -4.2628155, -6.0821409, -4.2628155, -1.3679767, 1.3679764
5: -2.9917998, -1.0055847, -2.9917998, -1.0055847, -1.3606665, 1.3606665
6: 2.2023408, 3.8499389, 2.2023408, 3.8499389, -1.4653525, 1.4653523
7: -10.2860727, -8.0480108, -10.2860727, -8.0480108, -1.5407121, 1.5407121
8: -1.9858670, 0.7323687, -1.9858670, 0.7323687, -1.8858261, 1.8858254
9: -8.5970411, -6.9790187, -8.5970411, -6.9790187, -1.3020208, 1.3020210

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4447196, upper bound: 0.4466220
time: 4.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4447233, upper bound: 0.4468670
time: 5.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.80 seconds
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4416839, upper bound: 0.4448137
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4416840, upper bound: 0.4450606
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4416840, upper bound: 0.4460800
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4416838, upper bound: 0.4463274
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4416838, upper bound: 0.4448172
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4416837, upper bound: 0.4450634
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4429493, upper bound: 0.4448135
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4429493, upper bound: 0.4450619
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4466069
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4468509
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4476797
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4436867, upper bound: 0.4479237
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4436865, upper bound: 0.4466100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4436864, upper bound: 0.4468514
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4447196, upper bound: 0.4466220
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 6, lower bound: -0.4447233, upper bound: 0.4468670

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.8082199, -3.0605731, -4.7966504, -3.0831509, -1.0825987, 1.0928836
1: -2.8655190, -0.8995788, -2.8501348, -0.9041888, -1.3317971, 1.3217256
2: -4.2109871, -2.3620057, -4.1936646, -2.3901048, -1.4059358, 1.4139755
3: -12.6480999, -9.9040365, -12.6238394, -9.9422493, -1.7723370, 1.7686000
4: -6.0675392, -4.2930446, -6.0342088, -4.3145967, -1.3153498, 1.3025198
5: -2.8946488, -1.0190573, -2.8191733, -1.0620191, -1.3144944, 1.3143533
6: 2.2131863, 3.8314745, 2.2763708, 3.7948182, -1.4274874, 1.4217744
7: -10.2686729, -8.1300478, -10.2257242, -8.1986351, -1.5079787, 1.5026116
8: -1.9177270, 0.7206547, -1.9028614, 0.7091815, -1.8194237, 1.8149254
9: -8.4982109, -6.9978375, -8.4884205, -7.0024433, -1.2554297, 1.2508717

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414423, upper bound: 0.4448143
time: 4.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414423, upper bound: 0.4448139
time: 4.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8179140, -3.0501313, -4.7967215, -3.0831401, -1.0957923, 1.1003308
1: -2.8921702, -0.8971381, -2.8501711, -0.9040251, -1.3425822, 1.3240194
2: -4.2146249, -2.3567672, -4.1936893, -2.3900726, -1.4112530, 1.4196048
3: -12.6504774, -9.8857279, -12.6238861, -9.9421616, -1.7752171, 1.7784672
4: -6.0813894, -4.2781978, -6.0342326, -4.3145418, -1.3244276, 1.3199229
5: -2.9047756, -1.0146480, -2.8191891, -1.0619559, -1.3172414, 1.3191345
6: 2.2054281, 3.8335209, 2.2763314, 3.7948277, -1.4317856, 1.4231789
7: -10.2848492, -8.1284571, -10.2257404, -8.1984568, -1.5106397, 1.5042055
8: -1.9319921, 0.7229507, -1.9029369, 0.7092326, -1.8371429, 1.8236756
9: -8.5184212, -6.9944730, -8.4884377, -7.0022893, -1.2647772, 1.2564845

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414422, upper bound: 0.4450565
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414423, upper bound: 0.4450631
time: 4.17 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.8095927, -3.0604525, -4.8531556, -3.0726755, -1.0945511, 1.0987902
1: -2.8658669, -0.8984449, -2.8640671, -0.8721727, -1.3380678, 1.3362014
2: -4.2121134, -2.3602242, -4.2013922, -2.3659923, -1.4185681, 1.4229679
3: -12.6485462, -9.9026585, -12.6300268, -9.9061975, -1.7795997, 1.7752450
4: -6.0693798, -4.2894154, -6.0450954, -4.2902269, -1.3282893, 1.3176169
5: -2.8980896, -1.0190070, -2.9101734, -1.0488610, -1.3236215, 1.3210468
6: 2.2123866, 3.8321533, 2.2675276, 3.8119941, -1.4337654, 1.4305723
7: -10.2691231, -8.1272497, -10.2421970, -8.1223488, -1.5122511, 1.5133171
8: -1.9207821, 0.7207370, -1.9646616, 0.7204924, -1.8342552, 1.8444774
9: -8.5016212, -6.9970307, -8.5816040, -6.9858160, -1.2700381, 1.2650018

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414421, upper bound: 0.4460798
time: 4.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414421, upper bound: 0.4460828
time: 4.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8192859, -3.0500116, -4.8532262, -3.0726650, -1.1032119, 1.1033618
1: -2.8925192, -0.8960040, -2.8641019, -0.8720101, -1.3447802, 1.3384948
2: -4.2157512, -2.3549860, -4.2014170, -2.3659608, -1.4224410, 1.4285963
3: -12.6509228, -9.8843517, -12.6300745, -9.9061098, -1.7824802, 1.7851157
4: -6.0832286, -4.2745686, -6.0451193, -4.2901726, -1.3352642, 1.3336129
5: -2.9082179, -1.0145984, -2.9101894, -1.0487981, -1.3263688, 1.3258282
6: 2.2046294, 3.8341992, 2.2674878, 3.8120027, -1.4380629, 1.4319789
7: -10.2852993, -8.1256599, -10.2422123, -8.1221733, -1.5149143, 1.5149114
8: -1.9350469, 0.7230332, -1.9647374, 0.7205443, -1.8519769, 1.8480144
9: -8.5218315, -6.9936676, -8.5816221, -6.9856639, -1.2769628, 1.2675433

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414421, upper bound: 0.4463218
time: 4.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414420, upper bound: 0.4463280
time: 4.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.8619633, -3.0501280, -4.7966504, -3.0831509, -1.0902205, 1.1016221
1: -2.8785791, -0.8700570, -2.8501348, -0.9041888, -1.3443727, 1.3266231
2: -4.2170949, -2.3412609, -4.1936646, -2.3901048, -1.4131575, 1.4191856
3: -12.6536264, -9.8705368, -12.6238394, -9.9422493, -1.7761555, 1.7732806
4: -6.0768347, -4.2774639, -6.0342088, -4.3145967, -1.3245375, 1.3157090
5: -2.9786754, -1.0060530, -2.8191733, -1.0620191, -1.3181386, 1.3200271
6: 2.2047377, 3.8470476, 2.2763708, 3.7948182, -1.4345999, 1.4256394
7: -10.2842293, -8.0597105, -10.2257242, -8.1986351, -1.5155599, 1.5036669
8: -1.9724784, 0.7318237, -1.9028614, 0.7091815, -1.8504462, 1.8260367
9: -8.5830812, -6.9822068, -8.4884205, -7.0024433, -1.2609224, 1.2629604

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4448172
time: 4.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4448156
time: 8.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8716602, -3.0396738, -4.7967215, -3.0831401, -1.0971923, 1.1062090
1: -2.9052298, -0.8676229, -2.8501711, -0.9040251, -1.3519239, 1.3288980
2: -4.2207403, -2.3360145, -4.1936893, -2.3900726, -1.4184542, 1.4248059
3: -12.6560078, -9.8522358, -12.6238861, -9.9421616, -1.7790427, 1.7831962
4: -6.0906973, -4.2626190, -6.0342326, -4.3145418, -1.3321867, 1.3228190
5: -2.9888005, -1.0016413, -2.8191891, -1.0619559, -1.3208890, 1.3248028
6: 2.1969752, 3.8490832, 2.2763314, 3.7948277, -1.4388890, 1.4270577
7: -10.3004007, -8.0581245, -10.2257404, -8.1984568, -1.5182538, 1.5052664
8: -1.9867544, 0.7341321, -1.9029369, 0.7092326, -1.8581538, 1.8347721
9: -8.6032791, -6.9788389, -8.4884377, -7.0022893, -1.2678595, 1.2655058

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4450592
time: 4.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4416893
time: 9.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8657041, -3.0498295, -4.8542867, -3.0726032, -1.1005502, 1.1067438
1: -2.8795187, -0.8662373, -2.8642943, -0.8708196, -1.3509965, 1.3419960
2: -4.2209015, -2.3361270, -4.2023711, -2.3648694, -1.4243870, 1.4322467
3: -12.6550369, -9.8665695, -12.6303368, -9.9053173, -1.7857275, 1.7821240
4: -6.0800133, -4.2646132, -6.0453229, -4.2862258, -1.3473446, 1.3457668
5: -2.9890487, -1.0059452, -2.9126117, -1.0488415, -1.3330946, 1.3314500
6: 2.2034130, 3.8495202, 2.2674675, 3.8127694, -1.4443369, 1.4387870
7: -10.2856989, -8.0507812, -10.2426043, -8.1201458, -1.5249228, 1.5211315
8: -1.9832227, 0.7320738, -1.9685044, 0.7205598, -1.8699088, 1.8567162
9: -8.5954361, -6.9804707, -8.5855532, -6.9855599, -1.2865009, 1.2814560

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4453525
time: 4.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4453509
time: 4.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8754001, -3.0393775, -4.8543572, -3.0725911, -1.1075213, 1.1113331
1: -2.9061701, -0.8638030, -2.8643291, -0.8706567, -1.3577039, 1.3442702
2: -4.2245474, -2.3308811, -4.2023964, -2.3648381, -1.4296627, 1.4378662
3: -12.6574154, -9.8482761, -12.6303825, -9.9052296, -1.7886140, 1.7920330
4: -6.0938735, -4.2497673, -6.0453472, -4.2861729, -1.3543613, 1.3528765
5: -2.9991736, -1.0015337, -2.9126275, -1.0487776, -1.3358452, 1.3362253
6: 2.1956520, 3.8515549, 2.2674270, 3.8127789, -1.4486182, 1.4402099
7: -10.3018684, -8.0491943, -10.2426205, -8.1199684, -1.5276284, 1.5227304
8: -1.9974966, 0.7343826, -1.9685779, 0.7206125, -1.8776140, 1.8602655
9: -8.6156282, -6.9771023, -8.5855703, -6.9854078, -1.2934413, 1.2840044

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4455940
time: 4.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4455991
time: 4.51 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.8100910, -3.0604291, -4.8103495, -3.0603852, -1.0995340, 1.0999665
1: -2.8659062, -0.8982424, -2.8660271, -0.8976312, -1.3431439, 1.3427794
2: -4.2121396, -2.3598800, -4.2122307, -2.3597655, -1.4121966, 1.4121164
3: -12.6485519, -9.9023142, -12.6487331, -9.9019775, -1.7720385, 1.7718639
4: -6.0698786, -4.2894087, -6.0699778, -4.2892118, -1.3175206, 1.3171968
5: -2.8987229, -1.0189874, -2.8987780, -1.0187507, -1.3428988, 1.3425267
6: 2.2121494, 3.8321543, 2.2120037, 3.8321874, -1.4450676, 1.4452767
7: -10.2691240, -8.1269789, -10.2691927, -8.1263409, -1.5226405, 1.5220716
8: -1.9212549, 0.7207525, -1.9215255, 0.7209172, -1.8354731, 1.8354049
9: -8.5016279, -6.9968853, -8.5017033, -6.9963589, -1.2751844, 1.2745266

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434480, upper bound: 0.4466074
time: 4.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434480, upper bound: 0.4466071
time: 5.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8197851, -3.0499883, -4.8104205, -3.0603743, -1.1140413, 1.1112871
1: -2.8925588, -0.8958018, -2.8660634, -0.8974690, -1.3546093, 1.3450329
2: -4.2157769, -2.3546414, -4.2122560, -2.3597350, -1.4175136, 1.4197582
3: -12.6509275, -9.8840084, -12.6487808, -9.9018898, -1.7748766, 1.7930193
4: -6.0837269, -4.2745624, -6.0700035, -4.2891579, -1.3329222, 1.3346004
5: -2.9088511, -1.0145772, -2.8987927, -1.0186875, -1.3456454, 1.3473086
6: 2.2043920, 3.8341994, 2.2119641, 3.8321984, -1.4525495, 1.4476876
7: -10.2853022, -8.1253872, -10.2692080, -8.1261635, -1.5253038, 1.5236664
8: -1.9355202, 0.7230494, -1.9216008, 0.7209687, -1.8531957, 1.8441513
9: -8.5218372, -6.9935222, -8.5017195, -6.9962063, -1.2821095, 1.2770805

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434481, upper bound: 0.4468461
time: 4.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434481, upper bound: 0.4468509
time: 4.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.8100910, -3.0604291, -4.8645430, -3.0499110, -1.1100752, 1.1100073
1: -2.8659062, -0.8982424, -2.8791163, -0.8677089, -1.3491952, 1.3553791
2: -4.2121396, -2.3598800, -4.2194977, -2.3388658, -1.4319644, 1.4206806
3: -12.6485519, -9.9023142, -12.6547394, -9.8683290, -1.7886696, 1.7785044
4: -6.0698786, -4.2894087, -6.0799289, -4.2705612, -1.3341043, 1.3271899
5: -2.8987229, -1.0189874, -2.9832726, -1.0057259, -1.3485711, 1.3475554
6: 2.2121494, 3.8321543, 2.2032669, 3.8481114, -1.4527681, 1.4546967
7: -10.2691240, -8.1269789, -10.2847948, -8.0553818, -1.5249734, 1.5296307
8: -1.9212549, 0.7207525, -1.9772186, 0.7321217, -1.8466158, 1.8697569
9: -8.5016279, -6.9968853, -8.5892010, -6.9805140, -1.2834425, 1.2811875

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4476824
time: 5.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4476788
time: 5.22 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8197851, -3.0499883, -4.8646131, -3.0498991, -1.1199327, 1.1145780
1: -2.8925588, -0.8958018, -2.8791533, -0.8675464, -1.3559043, 1.3576336
2: -4.2157769, -2.3546414, -4.2195230, -2.3388340, -1.4372771, 1.4283227
3: -12.6509275, -9.8840084, -12.6547852, -9.8682413, -1.7915483, 1.7973621
4: -6.0837269, -4.2745624, -6.0799541, -4.2705078, -1.3495059, 1.3445926
5: -2.9088511, -1.0145772, -2.9832890, -1.0056620, -1.3513176, 1.3523377
6: 2.2043920, 3.8341994, 2.2032270, 3.8481202, -1.4570637, 1.4571083
7: -10.2853022, -8.1253872, -10.2848101, -8.0552044, -1.5276375, 1.5312254
8: -1.9355202, 0.7230494, -1.9772935, 0.7321742, -1.8643398, 1.8732920
9: -8.5218372, -6.9935222, -8.5892172, -6.9803619, -1.2903674, 1.2837410

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4479209
time: 5.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4479235
time: 5.25 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.8642850, -3.0499544, -4.8103495, -3.0603852, -1.1095779, 1.1105065
1: -2.8789935, -0.8683207, -2.8660271, -0.8976312, -1.3557429, 1.3488351
2: -4.2194066, -2.3389769, -4.2122307, -2.3597655, -1.4207630, 1.4318851
3: -12.6545563, -9.8686686, -12.6487331, -9.9019775, -1.7786784, 1.7884812
4: -6.0798306, -4.2707591, -6.0699778, -4.2892118, -1.3275151, 1.3337803
5: -2.9832177, -1.0059619, -2.8987780, -1.0187507, -1.3479276, 1.3481987
6: 2.2034132, 3.8480759, 2.2120037, 3.8321874, -1.4544854, 1.4529676
7: -10.2847271, -8.0560207, -10.2691927, -8.1263409, -1.5301993, 1.5244057
8: -1.9769490, 0.7319562, -1.9215255, 0.7209172, -1.8698111, 1.8465474
9: -8.5891228, -6.9810410, -8.5017033, -6.9963589, -1.2818453, 1.2827847

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4466076
time: 4.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4466067
time: 5.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8739815, -3.0395014, -4.8104205, -3.0603743, -1.1165490, 1.1179726
1: -2.9056447, -0.8658864, -2.8660634, -0.8974690, -1.3638060, 1.3511082
2: -4.2230587, -2.3337302, -4.2122560, -2.3597350, -1.4260595, 1.4389277
3: -12.6569347, -9.8503723, -12.6487808, -9.9018898, -1.7815247, 1.7983971
4: -6.0936928, -4.2559161, -6.0700035, -4.2891579, -1.3429282, 1.3510647
5: -2.9933434, -1.0015504, -2.8987927, -1.0186875, -1.3506782, 1.3529744
6: 2.1956534, 3.8501103, 2.2119641, 3.8321984, -1.4600854, 1.4543891
7: -10.3008976, -8.0544319, -10.2692080, -8.1261635, -1.5328979, 1.5260057
8: -1.9912224, 0.7342649, -1.9216008, 0.7209687, -1.8775215, 1.8552830
9: -8.6093197, -6.9776726, -8.5017195, -6.9962063, -1.2887833, 1.2853432

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4468461
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4468514
time: 5.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8668251, -3.0497246, -4.8670821, -3.0496807, -1.1180134, 1.1184423
1: -2.8797693, -0.8654160, -2.8798919, -0.8648047, -1.3631070, 1.3627477
2: -4.2214012, -2.3347025, -4.2214923, -2.3345921, -1.4313672, 1.4312845
3: -12.6552019, -9.8654041, -12.6553822, -9.8650627, -1.7870283, 1.7868602
4: -6.0820112, -4.2630720, -6.0821095, -4.2628746, -1.3677499, 1.3674431
5: -2.9917264, -1.0058913, -2.9917812, -1.0056543, -1.3604615, 1.3600895
6: 2.2025335, 3.8498931, 2.2023869, 3.8499277, -1.4649048, 1.4651065
7: -10.2859850, -8.0488415, -10.2860527, -8.0482006, -1.5404422, 1.5398738
8: -1.9855108, 0.7321451, -1.9857826, 0.7323103, -1.8853426, 1.8852873
9: -8.5969419, -6.9797044, -8.5970192, -6.9791784, -1.3017540, 1.3010957

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4466213
time: 7.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4466213
time: 5.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8765197, -3.0392718, -4.8671536, -3.0496702, -1.1249847, 1.1230321
1: -2.9064217, -0.8629817, -2.8799276, -0.8646419, -1.3698115, 1.3650216
2: -4.2250466, -2.3294573, -4.2215176, -2.3345599, -1.4366436, 1.4389269
3: -12.6575813, -9.8471136, -12.6554308, -9.8649759, -1.7898626, 1.8061135
4: -6.0958710, -4.2482252, -6.0821342, -4.2628207, -1.3747685, 1.3745528
5: -3.0018513, -1.0014796, -2.9917963, -1.0055914, -1.3632128, 1.3648648
6: 2.1947730, 3.8519254, 2.2023478, 3.8499370, -1.4691846, 1.4665322
7: -10.3021536, -8.0472517, -10.2860661, -8.0480251, -1.5431504, 1.5414732
8: -1.9997830, 0.7344565, -1.9858584, 0.7323625, -1.8930478, 1.8888350
9: -8.6171350, -6.9763365, -8.5970345, -6.9790258, -1.3086948, 1.3036556

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4468601
time: 6.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4468649
time: 4.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.93 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414423, upper bound: 0.4448143
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414423, upper bound: 0.4448139
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414422, upper bound: 0.4450565
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414423, upper bound: 0.4450631
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414421, upper bound: 0.4460798
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414421, upper bound: 0.4460828
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414421, upper bound: 0.4463218
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414420, upper bound: 0.4463280
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4448172
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4448156
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4450592
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4414418, upper bound: 0.4416893
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4453525
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4453509
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4455940
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4430524, upper bound: 0.4455991
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434480, upper bound: 0.4466074
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434480, upper bound: 0.4466071
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434481, upper bound: 0.4468461
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434481, upper bound: 0.4468509
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4476824
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4476788
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4479209
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4434456, upper bound: 0.4479235
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4466076
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4466067
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4468461
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444781, upper bound: 0.4468514
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4466213
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4466213
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4468601
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.93
Output dim: 6, lower bound: -0.4444785, upper bound: 0.4468649
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.4518237113952637
rel_dist={6: [-0.44793373691976335, 0.4479324567017895]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 2410.72 seconds
