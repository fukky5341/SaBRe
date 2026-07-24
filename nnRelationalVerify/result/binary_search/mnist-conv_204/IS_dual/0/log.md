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
execution time: IAR + LP analysis = 15.27 + 32.06 = 47.33 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.67 seconds, max iter: 100)

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
Binary search time: 148.90 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 3403.77 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 430
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8161732, upper bound: 0.7947355
time: 4.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8170167, upper bound: 0.8170155
time: 4.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.84
Output dim: 6, lower bound: -0.8161732, upper bound: 0.7947355
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.84
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

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947327, upper bound: 0.7947334
time: 4.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947352, upper bound: 0.7947336
time: 5.71 seconds

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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

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
time: 4.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.13
Output dim: 6, lower bound: -0.7947327, upper bound: 0.7947334
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.13
Output dim: 6, lower bound: -0.7947352, upper bound: 0.7947336
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.13
Output dim: 6, lower bound: -0.7947352, upper bound: 0.8161718
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.13
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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.7947142
time: 4.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947109, upper bound: 0.7947137
time: 4.08 seconds

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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947139, upper bound: 0.7914976
time: 4.01 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947139, upper bound: 0.7947134
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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8161543
time: 4.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947109, upper bound: 0.8161537
time: 4.08 seconds

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

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8169987
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947134, upper bound: 0.8170003
time: 4.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.13 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7914982, upper bound: 0.7947142
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7947109, upper bound: 0.7947137
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7947139, upper bound: 0.7914976
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7947139, upper bound: 0.7947134
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8161543
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7947109, upper bound: 0.8161537
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -0.7914982, upper bound: 0.8169987
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.13
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

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7914971
time: 3.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7947142
time: 4.36 seconds

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

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7914971
time: 4.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7947142
time: 4.20 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8103390, -3.0604093, -1.3481479, 1.3464777
1: -2.8572679, -0.9035015, -2.8659072, -0.8976109, -1.6155217, 1.6179177
2: -4.1984229, -2.3881874, -4.2122245, -2.3604386, -1.6487622, 1.6371639
3: -12.6284370, -9.9407396, -12.6487703, -9.9023247, -2.1652374, 2.1557422
4: -6.0398083, -4.3091712, -6.0694137, -4.2891703, -1.5328450, 1.5417285
5: -2.8217437, -1.0503111, -2.8976047, -1.0186930, -1.5701001, 1.5625052
6: 2.2715242, 3.7956080, 2.2122087, 3.8320680, -1.5605438, 1.5833993
7: -10.2367878, -8.1965809, -10.2689981, -8.1271305, -1.8341775, 1.8484640
8: -1.9051766, 0.7185178, -1.9210391, 0.7209513, -2.0870233, 2.1019258
9: -8.4931936, -6.9887705, -8.5016956, -6.9965315, -1.4308310, 1.4467924

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7914965
time: 4.66 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129298, upper bound: 0.7914971
time: 4.63 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.7981510, -3.0748725, -4.8644028, -3.0499449, -1.3644977, 1.3651705
1: -2.8572509, -0.9035026, -2.8787906, -0.8673958, -1.6286891, 1.6324081
2: -4.1984158, -2.3881903, -4.2202578, -2.3409860, -1.6599445, 1.6508732
3: -12.6284332, -9.9407406, -12.6549988, -9.8695803, -2.1704302, 2.1687002
4: -6.0397978, -4.3091764, -6.0792766, -4.2673349, -1.5944695, 1.5515776
5: -2.8217409, -1.0503213, -2.9801745, -1.0056906, -1.5855951, 1.5677645
6: 2.2715335, 3.7956076, 2.2035723, 3.8483105, -1.5767770, 1.5920353
7: -10.2367706, -8.1965818, -10.2844257, -8.0572243, -1.8368101, 1.8622327
8: -1.9051728, 0.7185006, -1.9774642, 0.7321558, -2.1014724, 2.1469793
9: -8.4931870, -6.9888024, -8.5924807, -6.9809403, -1.4523339, 1.4649537

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7947135
time: 5.03 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7947133
time: 4.93 seconds

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

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.8129288
time: 4.26 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914985, upper bound: 0.8161544
time: 4.48 seconds

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

Time for backsubstitution: 13.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.8129287
time: 4.13 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947155, upper bound: 0.8161543
time: 4.19 seconds

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

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8137748
time: 4.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8170001
time: 4.83 seconds

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

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8137745
time: 4.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8169998
time: 4.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.89 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7914971
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7914985, upper bound: 0.7947142
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7914971
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7947155, upper bound: 0.7947142
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7914965
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.8129298, upper bound: 0.7914971
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7947135
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.8129278, upper bound: 0.7947133
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7914985, upper bound: 0.8129288
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7914985, upper bound: 0.8161544
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7947155, upper bound: 0.8129287
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7947155, upper bound: 0.8161543
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8137748
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7951256, upper bound: 0.8170001
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 6, lower bound: -0.7983490, upper bound: 0.8137745
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.89
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

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914950
time: 4.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914907
time: 4.54 seconds

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

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947119
time: 4.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947075
time: 4.55 seconds

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947128, upper bound: 0.7914905
time: 4.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914898
time: 4.84 seconds

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

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947133, upper bound: 0.7927128
time: 4.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947065, upper bound: 0.7927122
time: 4.31 seconds

## BFS IS instance: IS_A1_B2_B1_A1

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

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7914912
time: 4.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7914912
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_B1_A2

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7914916
time: 4.53 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7914908
time: 4.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7947069
time: 4.84 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7947069
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_B2_A2

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

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7927118
time: 4.82 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7927114
time: 4.53 seconds

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

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129266
time: 4.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129222
time: 4.33 seconds

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

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161522
time: 4.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161478
time: 4.51 seconds

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

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947060, upper bound: 0.8129260
time: 4.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947084, upper bound: 0.8129215
time: 4.69 seconds

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

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135207
time: 5.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135165
time: 4.57 seconds

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

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137729
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137685
time: 4.79 seconds

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

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169982
time: 5.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169940
time: 4.73 seconds

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

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983468, upper bound: 0.8137691
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137688
time: 4.69 seconds

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

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142789
time: 4.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142745
time: 4.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.12 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914950
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7914907
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947119
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947075
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947128, upper bound: 0.7914905
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914898
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947133, upper bound: 0.7927128
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947065, upper bound: 0.7927122
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7914912
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7914912
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7914916
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7914908
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7947069
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7947069
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7927118
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7927114
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129266
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129222
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161522
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161478
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947060, upper bound: 0.8129260
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947084, upper bound: 0.8129215
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135207
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135165
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137729
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137685
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169982
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169940
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7983468, upper bound: 0.8137691
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137688
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.12
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142789
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.12
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

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914919
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
time: 5.57 seconds

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

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914921
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
time: 4.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.02 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 24.02
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914919
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.02
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.02
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914921
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.02
Output dim: 6, lower bound: -0.7914934, upper bound: 0.7914920
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947119
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7914933, upper bound: 0.7947075
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947128, upper bound: 0.7914905
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947060, upper bound: 0.7914898
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947133, upper bound: 0.7927128
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947065, upper bound: 0.7927122
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7914912
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7914912
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7914916
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7914908
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7947069
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7947069
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129256, upper bound: 0.7927118
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.8129213, upper bound: 0.7927114
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129266
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8129222
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161522
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7914933, upper bound: 0.8161478
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947060, upper bound: 0.8129260
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947084, upper bound: 0.8129215
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135207
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7947065, upper bound: 0.8135165
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137729
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8137685
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169982
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7951208, upper bound: 0.8169940
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7983468, upper bound: 0.8137691
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7983425, upper bound: 0.8137688
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 6, lower bound: -0.7983430, upper bound: 0.8142789
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.02
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
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039341, upper bound: 0.5937624
time: 4.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6051967, upper bound: 0.6051922
time: 4.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.73
Output dim: 6, lower bound: -0.6039341, upper bound: 0.5937624
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.73
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

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937635
time: 4.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937659
time: 5.18 seconds

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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6051809, upper bound: 0.6018591
time: 4.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6051783, upper bound: 0.6051756
time: 4.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.38 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937635
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 6, lower bound: -0.5937659, upper bound: 0.5937659
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 6, lower bound: -0.6051809, upper bound: 0.6018591
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 6, lower bound: -0.6051783, upper bound: 0.6051756

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

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904141, upper bound: 0.5937455
time: 4.49 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937478, upper bound: 0.5937454
time: 4.79 seconds

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

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5904141
time: 4.46 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5937454
time: 4.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8119607, -3.0521667, -4.8096037, -3.0751381, -1.1933098, 1.2158399
1: -2.8731353, -0.8966422, -2.8596101, -0.8989528, -1.4600937, 1.4519293
2: -4.2174830, -2.3589528, -4.2100973, -2.3840904, -1.5029011, 1.5281237
3: -12.6534758, -9.9013138, -12.6464100, -9.9393482, -1.9312541, 1.9541221
4: -6.0744662, -4.2827525, -6.0421653, -4.2915211, -1.4353104, 1.4057250
5: -2.8993649, -1.0070109, -2.8261871, -1.0201793, -1.4567847, 1.4380596
6: 2.2077093, 3.8329823, 2.2677155, 3.8304286, -1.5456891, 1.5087464
7: -10.2801075, -8.1256685, -10.2666616, -8.1941404, -1.6539910, 1.6747022
8: -1.9245744, 0.7302809, -1.9138391, 0.7194419, -1.9453263, 1.9447892
9: -8.5070210, -6.9831071, -8.4988518, -6.9988680, -1.3493831, 1.3504744

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6005951
time: 4.41 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6018635
time: 6.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8120508, -3.0521631, -4.8654375, -3.0644438, -1.2047575, 1.2237484
1: -2.8732169, -0.8965083, -2.8737507, -0.8666631, -1.4723806, 1.4659035
2: -4.2174988, -2.3584032, -4.2194471, -2.3606772, -1.5258465, 1.5396076
3: -12.6534786, -9.9009523, -12.6531143, -9.9038486, -1.9533792, 1.9635644
4: -6.0748634, -4.2827559, -6.0539742, -4.2673206, -1.4661026, 1.4180124
5: -2.9003072, -1.0070276, -2.9161794, -1.0069659, -1.4706147, 1.4558294
6: 2.2075481, 3.8330698, 2.2582123, 3.8477392, -1.5517035, 1.5245507
7: -10.2802191, -8.1249056, -10.2831936, -8.1181393, -1.6708870, 1.6903644
8: -1.9250095, 0.7302704, -1.9761641, 0.7308450, -1.9574618, 1.9895363
9: -8.5070324, -6.9829407, -8.5923405, -6.9820576, -1.3666530, 1.3696220

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6039140
time: 4.67 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6051804
time: 5.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.76 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5904141, upper bound: 0.5937455
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937478, upper bound: 0.5937454
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5904141
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5937454
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6005951
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6018635
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6039140
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 6, lower bound: -0.5937483, upper bound: 0.6051804

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

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5904108
time: 4.41 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5937456
time: 4.51 seconds

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

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5904109
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5937456
time: 4.16 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.7981524, -3.0748608, -4.8102908, -3.0604286, -1.1961901, 1.1944995
1: -2.8572679, -0.9035015, -2.8658204, -0.8976988, -1.4448028, 1.4471354
2: -4.1984229, -2.3881874, -4.2122040, -2.3608241, -1.5125127, 1.5014116
3: -12.6284370, -9.9407396, -12.6487617, -9.9025698, -1.9314620, 1.9282649
4: -6.0398083, -4.3091712, -6.0690913, -4.2891803, -1.4034591, 1.4120140
5: -2.8217437, -1.0503111, -2.8969593, -1.0187006, -1.4200404, 1.4233572
6: 2.2715242, 3.7956080, 2.2123456, 3.8319960, -1.5110559, 1.5071292
7: -10.2367878, -8.1965809, -10.2688828, -8.1276636, -1.6434257, 1.6446075
8: -1.9051766, 0.7185178, -1.9207284, 0.7209404, -1.9256792, 1.9402714
9: -8.4931936, -6.9887705, -8.5016785, -6.9967117, -1.3269181, 1.3429191

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5904097
time: 8.11 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005954, upper bound: 0.5904120
time: 5.17 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.7981501, -3.0748796, -4.8636122, -3.0500102, -1.2072816, 1.2068615
1: -2.8572373, -0.9035025, -2.8786521, -0.8684112, -1.4530852, 1.4597161
2: -4.1984124, -2.3881917, -4.2193975, -2.3418810, -1.5195956, 1.5103843
3: -12.6284304, -9.9407406, -12.6547337, -9.8703499, -1.9364183, 1.9359450
4: -6.0397892, -4.3091798, -6.0787740, -4.2706404, -1.4381096, 1.4216907
5: -2.8217397, -1.0503249, -2.9780521, -1.0057082, -1.4326613, 1.4281102
6: 2.2715416, 3.7956061, 2.2037659, 3.8476748, -1.5153015, 1.5173514
7: -10.2367601, -8.1965828, -10.2841387, -8.0591869, -1.6452482, 1.6583033
8: -1.9051707, 0.7184894, -1.9745858, 0.7320957, -1.9371204, 1.9767017
9: -8.4931850, -6.9888229, -8.5890121, -6.9810963, -1.3435938, 1.3546381

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5937447
time: 4.80 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5937454
time: 4.76 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.8113523, -3.0522523, -4.7967277, -3.0831375, -1.1875994, 1.2025599
1: -2.8728540, -0.8971285, -2.8501735, -0.9040058, -1.4531593, 1.4385729
2: -4.2174025, -2.3605614, -4.1936932, -2.3900690, -1.5030913, 1.5097574
3: -12.6534424, -9.9024763, -12.6238947, -9.9421434, -1.9304969, 1.9282105
4: -6.0729752, -4.2827849, -6.0342388, -4.3145370, -1.4116516, 1.4024251
5: -2.8965611, -1.0070426, -2.8191915, -1.0619493, -1.4122810, 1.4287677
6: 2.2083516, 3.8327761, 2.2763240, 3.7948284, -1.5090275, 1.5086541
7: -10.2797794, -8.1277618, -10.2257423, -8.1984425, -1.6552968, 1.6320479
8: -1.9231553, 0.7302363, -1.9029455, 0.7092378, -1.9335074, 1.9323263
9: -8.5069609, -6.9836779, -8.4884434, -7.0022831, -1.3332357, 1.3367629

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6005950
time: 4.38 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6005949
time: 4.78 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.8120556, -3.0521393, -4.8104277, -3.0603719, -1.2038951, 1.2102146
1: -2.8732462, -0.8965058, -2.8660669, -0.8974495, -1.4655876, 1.4591022
2: -4.2175093, -2.3583987, -4.2122607, -2.3597307, -1.5104299, 1.5082983
3: -12.6534863, -9.9009495, -12.6487885, -9.9018717, -1.9336212, 1.9316626
4: -6.0748878, -4.2827406, -6.0700092, -4.2891531, -1.4156141, 1.4157090
5: -2.9003108, -1.0070014, -2.8987956, -1.0186815, -1.4435959, 1.4527744
6: 2.2075305, 3.8330717, 2.2119584, 3.8321991, -1.5292900, 1.5273876
7: -10.2802534, -8.1249027, -10.2692127, -8.1261492, -1.6678796, 1.6578252
8: -1.9250176, 0.7302978, -1.9216106, 0.7209752, -1.9474044, 1.9528151
9: -8.5070438, -6.9828806, -8.5017262, -6.9961996, -1.3514776, 1.3627300

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6018601
time: 5.43 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6018607
time: 4.50 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4.8120508, -3.0521631, -4.8540006, -3.0726142, -1.1994722, 1.2104442
1: -2.8732169, -0.8965083, -2.8642433, -0.8710400, -1.4627504, 1.4526596
2: -4.2174988, -2.3584032, -4.2021599, -2.3652840, -1.5208087, 1.5237595
3: -12.6534786, -9.9009523, -12.6303148, -9.9055433, -1.9454646, 1.9380336
4: -6.0748634, -4.2827559, -6.0452814, -4.2871208, -1.4423835, 1.4141893
5: -2.9003072, -1.0070276, -2.9117403, -1.0487795, -1.4276042, 1.4358947
6: 2.2075481, 3.8330698, 2.2674437, 3.8125644, -1.5154982, 1.5199988
7: -10.2802191, -8.1249056, -10.2424774, -8.1207237, -1.6605937, 1.6490235
8: -1.9250095, 0.7302704, -1.9674389, 0.7205958, -1.9473805, 1.9697473
9: -8.5070324, -6.9829407, -8.5846310, -6.9855022, -1.3504605, 1.3546188

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6039142
time: 4.08 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6008804
time: 5.29 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -4.8120508, -3.0521631, -4.8657060, -3.0498028, -1.2151327, 1.2224016
1: -2.8732169, -0.8965083, -2.8794260, -0.8662313, -1.4744844, 1.4721370
2: -4.2174988, -2.3584032, -4.2205291, -2.3373466, -1.5320632, 1.5181936
3: -12.6534786, -9.9009523, -12.6551123, -9.8670330, -1.9552598, 1.9409845
4: -6.0748634, -4.2827559, -6.0808401, -4.2666621, -1.4602947, 1.4267082
5: -2.9003072, -1.0070276, -2.9864578, -1.0056307, -1.4568741, 1.4591104
6: 2.2075481, 3.8330698, 2.2028737, 3.8489413, -1.5392773, 1.5426445
7: -10.2802191, -8.1249056, -10.2852869, -8.0524158, -1.6720138, 1.6725450
8: -1.9250095, 0.7302704, -1.9810665, 0.7322578, -1.9589438, 1.9939663
9: -8.5070324, -6.9829407, -8.5932064, -6.9798717, -1.3684738, 1.3724341

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6051783
time: 5.07 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6020793
time: 4.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5904108
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904134, upper bound: 0.5937456
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5904109
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5937483, upper bound: 0.5937456
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5904097
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.6005954, upper bound: 0.5904120
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5937447
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.6005930, upper bound: 0.5937454
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6005950
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6005949
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6018601
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6018607
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6039142
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6008804
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6051783
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.66
Output dim: 6, lower bound: -0.5904133, upper bound: 0.6020793

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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901554, upper bound: 0.5904033
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901534, upper bound: 0.5904024
time: 4.49 seconds

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

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

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
time: 4.33 seconds

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

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5934891, upper bound: 0.5904029
time: 4.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5934871, upper bound: 0.5904030
time: 4.78 seconds

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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5906702
time: 4.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5909193
time: 4.45 seconds

## BFS IS instance: IS_A1_B2_B1_A1

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

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003371, upper bound: 0.5904030
time: 4.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003376, upper bound: 0.5904034
time: 6.12 seconds

## BFS IS instance: IS_A1_B2_B1_A2

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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003371, upper bound: 0.5904039
time: 6.83 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005853, upper bound: 0.5904066
time: 4.85 seconds

## BFS IS instance: IS_A1_B2_B2_A1

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

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003371, upper bound: 0.5937366
time: 4.51 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005853, upper bound: 0.5937367
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_B2_A2

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

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003394, upper bound: 0.5909191
time: 6.19 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6005853, upper bound: 0.5909222
time: 4.70 seconds

## BFS IS instance: IS_A2_B1_B1_A1

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

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6003392
time: 4.37 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6005872
time: 4.43 seconds

## BFS IS instance: IS_A2_B1_B1_A2

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

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6003419
time: 4.84 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6005901
time: 4.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1

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

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925887, upper bound: 0.6016054
time: 5.10 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925888, upper bound: 0.6018519
time: 4.54 seconds

## BFS IS instance: IS_A2_B1_B2_A2

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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5923421, upper bound: 0.6018560
time: 4.59 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925887, upper bound: 0.6018560
time: 5.19 seconds

## BFS IS instance: IS_A2_B2_B1_A1

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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6036583
time: 4.50 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6039065
time: 4.55 seconds

## BFS IS instance: IS_A2_B2_B1_A2

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

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6006257
time: 5.77 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6005879
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_B2_A1

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

Time for backsubstitution: 12.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925887, upper bound: 0.6049247
time: 4.84 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925888, upper bound: 0.6051689
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_B2_A2

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

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925912, upper bound: 0.6018381
time: 5.15 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925888, upper bound: 0.6020716
time: 4.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.69 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5901554, upper bound: 0.5904033
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5901534, upper bound: 0.5904024
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5934871
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904067, upper bound: 0.5937384
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5934891, upper bound: 0.5904029
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5934871, upper bound: 0.5904030
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5906702
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5937408, upper bound: 0.5909193
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6003371, upper bound: 0.5904030
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6003376, upper bound: 0.5904034
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6003371, upper bound: 0.5904039
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6005853, upper bound: 0.5904066
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6003371, upper bound: 0.5937366
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6005853, upper bound: 0.5937367
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6003394, upper bound: 0.5909191
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.6005853, upper bound: 0.5909222
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6003392
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6005872
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6003419
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6005901
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925887, upper bound: 0.6016054
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925888, upper bound: 0.6018519
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5923421, upper bound: 0.6018560
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925887, upper bound: 0.6018560
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6036583
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6039065
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6006257
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5904059, upper bound: 0.6005879
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925887, upper bound: 0.6049247
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925888, upper bound: 0.6051689
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925912, upper bound: 0.6018381
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 6, lower bound: -0.5925888, upper bound: 0.6020716

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.7967277, -3.0831375, -4.7963915, -3.0831957, -1.1724870, 1.1719265
1: -2.8501735, -0.9040058, -2.8500171, -0.9047991, -1.4304066, 1.4308805
2: -4.1936932, -2.3900690, -4.1935730, -2.3902187, -1.4786062, 1.4787110
3: -12.6238947, -9.9421434, -12.6236572, -9.9425879, -1.9001045, 1.9003336
4: -6.0342388, -4.3145370, -6.0341082, -4.3147926, -1.3676536, 1.3680739
5: -2.8191915, -1.0619493, -2.8191175, -1.0622549, -1.3898840, 1.3903632
6: 2.2763240, 3.7948284, 2.2765188, 3.7947831, -1.4732933, 1.4730196
7: -10.2257423, -8.1984425, -10.2256594, -8.1992712, -1.6171088, 1.6178608
8: -1.9029455, 0.7092378, -1.9025915, 0.7090173, -1.9111524, 1.9112310
9: -8.4884434, -7.0022831, -8.4883442, -7.0029688, -1.3146372, 1.3154981

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901536, upper bound: 0.5901508
time: 4.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5901536, upper bound: 0.5904042
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.7967238, -3.0831389, -4.8119717, -3.0716639, -1.1847715, 1.1976576
1: -2.8501732, -0.9040166, -2.8834500, -0.9021467, -1.4337416, 1.4503717
2: -4.1936903, -2.3900714, -4.1982145, -2.3841133, -1.4873905, 1.4854714
3: -12.6238890, -9.9421549, -12.6262789, -9.9178085, -1.9300165, 1.9038284
4: -6.0342355, -4.3145409, -6.0491982, -4.2887688, -1.3964119, 1.3849316
5: -2.8191900, -1.0619533, -2.8297834, -1.0534532, -1.4105213, 1.4008670
6: 2.2763290, 3.7948279, 2.2660160, 3.7968991, -1.4757071, 1.4842176
7: -10.2257404, -8.1984501, -10.2423162, -8.1975422, -1.6197414, 1.6300590
8: -1.9029408, 0.7092345, -1.9210150, 0.7117364, -1.9220686, 1.9361076
9: -8.4884396, -7.0022874, -8.5094090, -6.9976163, -1.3225689, 1.3303688

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904069, upper bound: 0.5901512
time: 4.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5904069, upper bound: 0.5904041
time: 4.66 seconds

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

Time for backsubstitution: 12.89 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.525653600692749
rel_dist={6: [-0.6052081522674859, 0.6052060697814263]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 430
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466822, upper bound: 0.4433082
time: 4.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479277, upper bound: 0.4479263
time: 4.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.03
Output dim: 6, lower bound: -0.4466822, upper bound: 0.4433082
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.03
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

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4463359, upper bound: 0.4416896
time: 4.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466781, upper bound: 0.4433000
time: 4.38 seconds

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

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6114
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6114

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479261, upper bound: 0.4468528
time: 4.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479261, upper bound: 0.4479242
time: 4.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.53
Output dim: 6, lower bound: -0.4463359, upper bound: 0.4416896
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.53
Output dim: 6, lower bound: -0.4466781, upper bound: 0.4433000
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.53
Output dim: 6, lower bound: -0.4479261, upper bound: 0.4468528
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.53
Output dim: 6, lower bound: -0.4479261, upper bound: 0.4479242

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.7973332, -3.0797892, -4.7988720, -3.0817492, -1.0731480, 1.0767708
1: -2.8530421, -0.9037985, -2.8517430, -0.9032216, -1.3213897, 1.3199275
2: -4.1955795, -2.3892670, -4.1964388, -2.3889525, -1.3913171, 1.3925092
3: -12.6257229, -9.9415379, -12.6277037, -9.9415817, -1.7518983, 1.7539735
4: -6.0364904, -4.3123026, -6.0357304, -4.3107281, -1.2896895, 1.2872260
5: -2.8203011, -1.0572393, -2.8205628, -1.0548596, -1.3100815, 1.3077919
6: 2.2743831, 3.7951503, 2.2747793, 3.8008575, -1.4079309, 1.4022160
7: -10.2302113, -8.1976471, -10.2326880, -8.1976185, -1.4972329, 1.4994791
8: -1.9038587, 0.7129884, -1.9046106, 0.7109611, -1.8068132, 1.8096743
9: -8.4904213, -6.9968147, -8.4901361, -7.0016875, -1.2499681, 1.2544894

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4429586, upper bound: 0.4416891
time: 4.53 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4429586, upper bound: 0.4416908
time: 5.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.7981491, -3.0748870, -4.8537250, -3.0713496, -1.0817528, 1.0984261
1: -2.8572197, -0.9035041, -2.8652654, -0.8727328, -1.3339236, 1.3324451
2: -4.1984062, -2.3881941, -4.2036538, -2.3672583, -1.4149215, 1.4009699
3: -12.6284285, -9.9407425, -12.6337204, -9.9072800, -1.7766426, 1.7610826
4: -6.0397758, -4.3091831, -6.0463190, -4.2894535, -1.3264561, 1.3029954
5: -2.8217375, -1.0503302, -2.9070799, -1.0417318, -1.3214371, 1.3263408
6: 2.2715530, 3.7956052, 2.2660162, 3.8172331, -1.4279163, 1.4164569
7: -10.2367487, -8.1965837, -10.2485113, -8.1249533, -1.5142291, 1.5104876
8: -1.9051695, 0.7184727, -1.9619675, 0.7221928, -1.8178387, 1.8498828
9: -8.4931822, -6.9888439, -8.5802803, -6.9855547, -1.2660418, 1.2745092

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4433037, upper bound: 0.4432997
time: 4.26 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4433037, upper bound: 0.4433010
time: 7.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8109722, -3.0570827, -4.8095994, -3.0751395, -1.0903258, 1.1096797
1: -2.8688030, -0.8972859, -2.8596067, -0.8989543, -1.3420382, 1.3365781
2: -4.2142191, -2.3599305, -4.2100906, -2.3840923, -1.4069080, 1.4342217
3: -12.6506386, -9.9019814, -12.6464005, -9.9393492, -1.7716441, 1.7969422
4: -6.0713620, -4.2869234, -6.0421629, -4.2915282, -1.3432503, 1.3113742
5: -2.8981271, -1.0139642, -2.8261852, -1.0201924, -1.3625007, 1.3369267
6: 2.2104383, 3.8323872, 2.2677183, 3.8304172, -1.4672554, 1.4280472
7: -10.2734528, -8.1267138, -10.2666512, -8.1941414, -1.5148587, 1.5468724
8: -1.9222465, 0.7247171, -1.9138343, 0.7194369, -1.8351202, 1.8316913
9: -8.5036907, -6.9911547, -8.4988480, -6.9988689, -1.2748837, 1.2724853

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476827, upper bound: 0.4468446
time: 4.44 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476821, upper bound: 0.4468508
time: 4.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8120499, -3.0521722, -4.8648391, -3.0644851, -1.0993483, 1.1205516
1: -2.8731987, -0.8965095, -2.8736391, -0.8674289, -1.3560514, 1.3496807
2: -4.2174945, -2.3584051, -4.2188048, -2.3612170, -1.4324415, 1.4450386
3: -12.6534758, -9.9009533, -12.6529074, -9.9043112, -1.7921801, 1.8043628
4: -6.0748482, -4.2827592, -6.0537577, -4.2699265, -1.3651111, 1.3293085
5: -2.9003053, -1.0070348, -2.9148812, -1.0069945, -1.3743556, 1.3574587
6: 2.2075586, 3.8330688, 2.2582812, 3.8472545, -1.4762073, 1.4428422
7: -10.2802029, -8.1249065, -10.2829666, -8.1193533, -1.5361812, 1.5595381
8: -1.9250052, 0.7302551, -1.9739895, 0.7308040, -1.8478432, 1.8763993
9: -8.5070286, -6.9829683, -8.5897522, -6.9822021, -1.2917428, 1.2963936

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6114
type: B, layer: 1, pos: 430
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4468554, upper bound: 0.4479251
time: 4.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4468556, upper bound: 0.4479274
time: 4.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.63 seconds
IS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4429586, upper bound: 0.4416891
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4429586, upper bound: 0.4416908
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4433037, upper bound: 0.4432997
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4433037, upper bound: 0.4433010
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4476827, upper bound: 0.4468446
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4476821, upper bound: 0.4468508
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4468554, upper bound: 0.4479251
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.63
Output dim: 6, lower bound: -0.4468556, upper bound: 0.4479274

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.8108954, -3.0570955, -4.8092642, -3.0751979, -1.0901272, 1.1090493
1: -2.8687632, -0.8974683, -2.8594475, -0.8997489, -1.3411772, 1.3360853
2: -4.2141891, -2.3599658, -4.2099700, -2.3842425, -1.4066327, 1.4340329
3: -12.6505833, -9.9020872, -12.6461639, -9.9397907, -1.7711506, 1.7966170
4: -6.0713310, -4.2869830, -6.0420313, -4.2917852, -1.3427029, 1.3111503
5: -2.8981090, -1.0140340, -2.8261125, -1.0204983, -1.3619232, 1.3367307
6: 2.2104821, 3.8323767, 2.2679093, 3.8303714, -1.4670098, 1.4275978
7: -10.2734337, -8.1269054, -10.2665653, -8.1949692, -1.5140235, 1.5466018
8: -1.9221618, 0.7246604, -1.9134812, 0.7192168, -1.8346066, 1.8312449
9: -8.5036688, -6.9913139, -8.4987488, -6.9995556, -1.2739587, 1.2722256

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476796, upper bound: 0.4466056
time: 4.84 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476796, upper bound: 0.4468442
time: 5.09 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.8109646, -3.0570846, -4.8189545, -3.0647078, -1.1014915, 1.1187370
1: -2.8687990, -0.8973061, -2.8861046, -0.8973234, -1.3434374, 1.3477403
2: -4.2142143, -2.3599355, -4.2135997, -2.3790166, -1.4142365, 1.4379015
3: -12.6506290, -9.9019985, -12.6485338, -9.9214849, -1.7923696, 1.7994862
4: -6.0713558, -4.2869291, -6.0559301, -4.2769413, -1.3527141, 1.3266058
5: -2.8981247, -1.0139704, -2.8362401, -1.0160906, -1.3667169, 1.3449473
6: 2.2104430, 3.8323858, 2.2601633, 3.8324203, -1.4684095, 1.4370222
7: -10.2734489, -8.1267271, -10.2827387, -8.1933727, -1.5155973, 1.5492544
8: -1.9222364, 0.7247131, -1.9277403, 0.7215202, -1.8433509, 1.8488557
9: -8.5036850, -6.9911623, -8.5189619, -6.9961710, -1.2764969, 1.2833891

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479181, upper bound: 0.4466061
time: 4.53 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479206, upper bound: 0.4466061
time: 4.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8104277, -3.0603719, -4.8643651, -3.0645225, -1.1003559, 1.1122484
1: -2.8660669, -0.8974495, -2.8735096, -0.8679671, -1.3487341, 1.3500288
2: -4.2122607, -2.3597307, -4.2184572, -2.3618696, -1.4262524, 1.4408445
3: -12.6487885, -9.9018717, -12.6527920, -9.9048014, -1.7873294, 1.8018429
4: -6.0700092, -4.2891531, -6.0535507, -4.2711182, -1.3525887, 1.3195965
5: -2.8987956, -1.0186815, -2.9135771, -1.0070062, -1.3677130, 1.3451254
6: 2.2119584, 3.8321991, 2.2583590, 3.8469431, -1.4705935, 1.4372318
7: -10.2692127, -8.1261492, -10.2827606, -8.1204729, -1.5242109, 1.5552006
8: -1.9216106, 0.7209752, -1.9724393, 0.7307711, -1.8455849, 1.8660955
9: -8.5017262, -6.9961996, -8.5883675, -6.9823761, -1.2820101, 1.2807364

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466090, upper bound: 0.4479147
time: 4.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466092, upper bound: 0.4479213
time: 4.93 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8670249, -3.0496831, -4.8659668, -3.0644031, -1.1150637, 1.1203711
1: -2.8798587, -0.8647368, -2.8739119, -0.8660870, -1.3620017, 1.3569726
2: -4.2215066, -2.3349638, -4.2197666, -2.3598418, -1.4241595, 1.4517186
3: -12.6554327, -9.8652372, -12.6532125, -9.9032402, -1.7849321, 1.8102121
4: -6.0819683, -4.2628241, -6.0542197, -4.2660570, -1.3740883, 1.3609829
5: -2.9911106, -1.0055921, -2.9177754, -1.0069704, -1.3794086, 1.3564231
6: 2.2024193, 3.8498755, 2.2581160, 3.8480506, -1.4820893, 1.4484191
7: -10.2859735, -8.0485325, -10.2834291, -8.1168051, -1.5380192, 1.5649276
8: -1.9855015, 0.7323573, -1.9778337, 0.7308772, -1.8840303, 1.8793883
9: -8.5970259, -6.9791441, -8.5936031, -6.9818363, -1.2999647, 1.2987010

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434328, upper bound: 0.4467101
time: 5.19 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4468548, upper bound: 0.4468674
time: 4.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 47.49 seconds
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4476796, upper bound: 0.4466056
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4476796, upper bound: 0.4468442
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4479181, upper bound: 0.4466061
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4479206, upper bound: 0.4466061
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4466090, upper bound: 0.4479147
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4466092, upper bound: 0.4479213
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4434328, upper bound: 0.4467101
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 47.49
Output dim: 6, lower bound: -0.4468548, upper bound: 0.4468674

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -4.8106365, -3.0571394, -4.8092642, -3.0751979, -1.0896521, 1.1090062
1: -2.8686416, -0.8980802, -2.8594475, -0.8997489, -1.3409369, 1.3354800
2: -4.2140980, -2.3600798, -4.2099700, -2.3842425, -1.4065316, 1.4338484
3: -12.6504002, -9.9024258, -12.6461639, -9.9397907, -1.7709861, 1.7962596
4: -6.0712299, -4.2871809, -6.0420313, -4.2917852, -1.3426242, 1.3107474
5: -2.8980539, -1.0142705, -2.8261125, -1.0204983, -1.3618684, 1.3363070
6: 2.2106278, 3.8323424, 2.2679093, 3.8303714, -1.4666955, 1.4274890
7: -10.2733660, -8.1275425, -10.2665653, -8.1949692, -1.5139720, 1.5459774
8: -1.9218905, 0.7244956, -1.9134812, 0.7192168, -1.8343410, 1.8309131
9: -8.5035925, -6.9918399, -8.4987488, -6.9995556, -1.2739205, 1.2715242

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4463839
time: 4.83 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476801, upper bound: 0.4466087
time: 4.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8195848, -3.0468266, -4.8092642, -3.0751979, -1.1000018, 1.1159918
1: -2.8944075, -0.8956710, -2.8594475, -0.8997489, -1.3539765, 1.3377547
2: -4.2176023, -2.3549871, -4.2099700, -2.3842425, -1.4115994, 1.4392550
3: -12.6527424, -9.8849792, -12.6461639, -9.9397907, -1.7736411, 1.8053122
4: -6.0849018, -4.2737556, -6.0420313, -4.2917852, -1.3521476, 1.3235497
5: -2.9080899, -1.0104105, -2.8261125, -1.0204983, -1.3644145, 1.3421049
6: 2.2032745, 3.8343663, 2.2679093, 3.8303714, -1.4706690, 1.4297807
7: -10.2894611, -8.1259861, -10.2665653, -8.1949692, -1.5238938, 1.5474513
8: -1.9355686, 0.7267044, -1.9134812, 0.7192168, -1.8509984, 1.8342342
9: -8.5236454, -6.9887276, -8.4987488, -6.9995556, -1.2803359, 1.2739935

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4466233
time: 4.66 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476801, upper bound: 0.4468448
time: 4.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4.8106365, -3.0571394, -4.8182116, -3.0648398, -1.1007271, 1.1176896
1: -2.8686416, -0.8980802, -2.8852184, -0.8973542, -1.3432183, 1.3464572
2: -4.2140980, -2.3600798, -4.2134671, -2.3791628, -1.4127097, 1.4375119
3: -12.6504002, -9.9024258, -12.6484985, -9.9223461, -1.7913165, 1.7989023
4: -6.0712299, -4.2871809, -6.0557480, -4.2783642, -1.3515050, 1.3258395
5: -2.8980539, -1.0142705, -2.8361483, -1.0166399, -1.3656228, 1.3443196
6: 2.2106278, 3.8323424, 2.2605660, 3.8323979, -1.4679754, 1.4340072
7: -10.2733660, -8.1275425, -10.2826567, -8.1934042, -1.5155919, 1.5483301
8: -1.9218905, 0.7244956, -1.9271560, 0.7214296, -1.8376617, 1.8475604
9: -8.5035925, -6.9918399, -8.5188046, -6.9964218, -1.2762346, 1.2821851

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443529, upper bound: 0.4463844
time: 5.11 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476795, upper bound: 0.4466054
time: 5.12 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8278708, -3.0454879, -4.8264976, -3.0634727, -1.1232345, 1.1315763
1: -2.9038637, -0.8953234, -2.8946767, -0.8970169, -1.3616753, 1.3541995
2: -4.2190580, -2.3537931, -4.2149162, -2.3779709, -1.4214237, 1.4458771
3: -12.6531258, -9.8759584, -12.6488714, -9.9133253, -1.8038073, 1.8177154
4: -6.0864534, -4.2580128, -6.0573201, -4.2626200, -1.3728058, 1.3566294
5: -2.9087782, -1.0042229, -2.8368413, -1.0104530, -1.3772027, 1.3570838
6: 2.1995416, 3.8344662, 2.2568297, 3.8324988, -1.4738193, 1.4403298
7: -10.2900724, -8.1258163, -10.2832785, -8.1932316, -1.5244365, 1.5510917
8: -1.9414380, 0.7272565, -1.9330204, 0.7219806, -1.8663912, 1.8628182
9: -8.5248051, -6.9859419, -8.5199652, -6.9936194, -1.2857335, 1.2876112

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443529, upper bound: 0.4463841
time: 5.07 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476795, upper bound: 0.4466060
time: 4.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.8103495, -3.0603852, -4.8640304, -3.0645807, -1.1001558, 1.1116176
1: -2.8660271, -0.8976312, -2.8733499, -0.8687603, -1.3478725, 1.3495347
2: -4.2122307, -2.3597655, -4.2183371, -2.3620181, -1.4259789, 1.4406557
3: -12.6487331, -9.9019775, -12.6525564, -9.9052448, -1.7868121, 1.8015175
4: -6.0699778, -4.2892118, -6.0534210, -4.2713747, -1.3520541, 1.3193736
5: -2.8987780, -1.0187507, -2.9135041, -1.0073123, -1.3671367, 1.3449218
6: 2.2120037, 3.8321874, 2.2585521, 3.8468966, -1.4703476, 1.4367805
7: -10.2691927, -8.1263409, -10.2826748, -8.1213026, -1.5233724, 1.5549316
8: -1.9215255, 0.7209172, -1.9720848, 0.7305484, -1.8450713, 1.8656154
9: -8.5017033, -6.9963589, -8.5882683, -6.9830627, -1.2810850, 1.2804692

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466098, upper bound: 0.4476783
time: 4.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466098, upper bound: 0.4479165
time: 4.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8104205, -3.0603743, -4.8737259, -3.0540771, -1.1115355, 1.1185911
1: -2.8660634, -0.8974690, -2.9000039, -0.8663414, -1.3501277, 1.3574945
2: -4.2122560, -2.3597350, -4.2219906, -2.3567832, -1.4336033, 1.4445379
3: -12.6487808, -9.9018898, -12.6549301, -9.8869505, -1.7967157, 1.8043945
4: -6.0700035, -4.2891579, -6.0673323, -4.2565308, -1.3591671, 1.3348403
5: -2.8987927, -1.0186875, -2.9236293, -1.0029032, -1.3719234, 1.3476596
6: 2.2119641, 3.8321984, 2.2508023, 3.8489335, -1.4717629, 1.4462006
7: -10.2692080, -8.1261635, -10.2988491, -8.1197081, -1.5249579, 1.5576212
8: -1.9216008, 0.7209687, -1.9863534, 0.7328622, -1.8538094, 1.8732076
9: -8.5017195, -6.9962063, -8.6084652, -6.9796877, -1.2836299, 1.2874010

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4468484, upper bound: 0.4476782
time: 4.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4468484, upper bound: 0.4476755
time: 4.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8670230, -3.0496836, -4.8659658, -3.0644028, -1.1150627, 1.1203699
1: -2.8798583, -0.8647393, -2.8739116, -0.8660913, -1.3619971, 1.3569698
2: -4.2215061, -2.3349645, -4.2197657, -2.3598416, -1.4241571, 1.4517181
3: -12.6554289, -9.8652401, -12.6532116, -9.9032402, -1.7849312, 1.8102095
4: -6.0819683, -4.2628269, -6.0542197, -4.2660618, -1.3740833, 1.3609788
5: -2.9911103, -1.0055921, -2.9177754, -1.0069704, -1.3794072, 1.3564197
6: 2.2024195, 3.8498731, 2.2581170, 3.8480458, -1.4820838, 1.4484167
7: -10.2859716, -8.0485325, -10.2834244, -8.1168051, -1.5380168, 1.5649238
8: -1.9855003, 0.7323568, -1.9778323, 0.7308764, -1.8840294, 1.8793888
9: -8.5970240, -6.9791446, -8.5935984, -6.9818344, -1.2999620, 1.2986968

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443534, upper bound: 0.4467050
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445982, upper bound: 0.4467111
time: 4.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8670244, -3.0496833, -4.8659687, -3.0644035, -1.1150637, 1.1203711
1: -2.8798587, -0.8647385, -2.8739076, -0.8660884, -1.3619993, 1.3569729
2: -4.2215061, -2.3349731, -4.2197652, -2.3598413, -1.4241562, 1.4517186
3: -12.6554279, -9.8652382, -12.6532135, -9.9032116, -1.7849102, 1.8102105
4: -6.0819674, -4.2628250, -6.0542545, -4.2660565, -1.3740842, 1.3609827
5: -2.9910955, -1.0055919, -2.9177666, -1.0069730, -1.3794081, 1.3564215
6: 2.2024202, 3.8498752, 2.2580934, 3.8480496, -1.4820855, 1.4484627
7: -10.2859735, -8.0485325, -10.2834291, -8.1167679, -1.5380192, 1.5649252
8: -1.9854999, 0.7323580, -1.9778333, 0.7308757, -1.8840294, 1.8793893
9: -8.5970259, -6.9791441, -8.5936022, -6.9818316, -1.2999649, 1.2986984

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4476800, upper bound: 0.4468589
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4479242, upper bound: 0.4468659
time: 4.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.04 seconds
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4463839
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4476801, upper bound: 0.4466087
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4466233
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4476801, upper bound: 0.4468448
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4443529, upper bound: 0.4463844
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4476795, upper bound: 0.4466054
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4443529, upper bound: 0.4463841
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4476795, upper bound: 0.4466060
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4466098, upper bound: 0.4476783
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4466098, upper bound: 0.4479165
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4468484, upper bound: 0.4476782
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4468484, upper bound: 0.4476755
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4443534, upper bound: 0.4467050
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4445982, upper bound: 0.4467111
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4476800, upper bound: 0.4468589
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 6, lower bound: -0.4479242, upper bound: 0.4468659

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.8106356, -3.0571394, -4.8092623, -3.0751972, -1.0896509, 1.1090047
1: -2.8686419, -0.8980826, -2.8594468, -0.8997531, -1.3409340, 1.3354795
2: -4.2140985, -2.3600798, -4.2099695, -2.3842437, -1.4065299, 1.4338472
3: -12.6503983, -9.9024258, -12.6461611, -9.9397926, -1.7709856, 1.7962568
4: -6.0712299, -4.2871838, -6.0420308, -4.2917905, -1.3426185, 1.3107445
5: -2.8980551, -1.0142703, -2.8261094, -1.0204983, -1.3618665, 1.3363028
6: 2.2106280, 3.8323390, 2.2679100, 3.8303657, -1.4666903, 1.4274859
7: -10.2733612, -8.1275434, -10.2665615, -8.1949711, -1.5139680, 1.5459728
8: -1.9218886, 0.7244959, -1.9134808, 0.7192156, -1.8343420, 1.8309135
9: -8.5035896, -6.9918408, -8.4987450, -6.9995561, -1.2739177, 1.2715194

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443544, upper bound: 0.4431830
time: 4.78 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443544, upper bound: 0.4463894
time: 4.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.8106360, -3.0571396, -4.8092637, -3.0751944, -1.0896621, 1.1090057
1: -2.8686416, -0.8980811, -2.8594449, -0.8997483, -1.3409343, 1.3354530
2: -4.2140980, -2.3600872, -4.2099686, -2.3842430, -1.4065287, 1.4338479
3: -12.6504002, -9.9024258, -12.6461649, -9.9397688, -1.7709651, 1.7962575
4: -6.0712285, -4.2871799, -6.0420713, -4.2917852, -1.3426199, 1.3107855
5: -2.8980389, -1.0142703, -2.8261118, -1.0205007, -1.3618677, 1.3363020
6: 2.2106287, 3.8323410, 2.2678821, 3.8303704, -1.4666927, 1.4275312
7: -10.2733631, -8.1275425, -10.2665653, -8.1949434, -1.5140052, 1.5459754
8: -1.9218903, 0.7244949, -1.9134824, 0.7192163, -1.8343420, 1.8309140
9: -8.5035915, -6.9918408, -8.4987488, -6.9995584, -1.2739201, 1.2715204

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4475317, upper bound: 0.4431858
time: 4.86 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4475317, upper bound: 0.4431820
time: 5.00 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8195844, -3.0468273, -4.8092623, -3.0751972, -1.1000011, 1.1159909
1: -2.8944063, -0.8956732, -2.8594468, -0.8997531, -1.3539717, 1.3377545
2: -4.2176023, -2.3549871, -4.2099695, -2.3842437, -1.4115982, 1.4392538
3: -12.6527395, -9.8849792, -12.6461611, -9.9397926, -1.7736411, 1.8053086
4: -6.0849018, -4.2737594, -6.0420308, -4.2917905, -1.3521421, 1.3235471
5: -2.9080884, -1.0104105, -2.8261094, -1.0204983, -1.3644125, 1.3421013
6: 2.2032747, 3.8343635, 2.2679100, 3.8303657, -1.4706638, 1.4297774
7: -10.2894573, -8.1259861, -10.2665615, -8.1949711, -1.5238914, 1.5474465
8: -1.9355698, 0.7267039, -1.9134808, 0.7192156, -1.8509979, 1.8342338
9: -8.5236435, -6.9887285, -8.4987450, -6.9995561, -1.2803330, 1.2739892

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4434215
time: 4.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4466274
time: 4.96 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8195844, -3.0468271, -4.8092637, -3.0751944, -1.1000123, 1.1159916
1: -2.8944066, -0.8956707, -2.8594449, -0.8997483, -1.3539739, 1.3377275
2: -4.2176023, -2.3549957, -4.2099686, -2.3842430, -1.4115975, 1.4392548
3: -12.6527414, -9.8849802, -12.6461649, -9.9397688, -1.7736211, 1.8053093
4: -6.0849009, -4.2737560, -6.0420713, -4.2917852, -1.3521430, 1.3235881
5: -2.9080739, -1.0104103, -2.8261118, -1.0205007, -1.3644135, 1.3421001
6: 2.2032747, 3.8343668, 2.2678821, 3.8303704, -1.4706662, 1.4298229
7: -10.2894611, -8.1259861, -10.2665653, -8.1949434, -1.5238948, 1.5474484
8: -1.9355688, 0.7267051, -1.9134824, 0.7192163, -1.8509979, 1.8342347
9: -8.5236454, -6.9887285, -8.4987488, -6.9995584, -1.2803364, 1.2739897

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4475308, upper bound: 0.4434243
time: 5.01 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4475308, upper bound: 0.4468425
time: 5.22 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.8106356, -3.0571394, -4.8182101, -3.0648408, -1.1007252, 1.1176877
1: -2.8686419, -0.8980826, -2.8852184, -0.8973587, -1.3432145, 1.3464543
2: -4.2140985, -2.3600798, -4.2134666, -2.3791645, -1.4127088, 1.4375107
3: -12.6503983, -9.9024258, -12.6484947, -9.9223442, -1.7913160, 1.7988997
4: -6.0712299, -4.2871838, -6.0557470, -4.2783689, -1.3514996, 1.3258371
5: -2.8980551, -1.0142703, -2.8361449, -1.0166399, -1.3656201, 1.3443165
6: 2.2106280, 3.8323390, 2.2605662, 3.8323917, -1.4679699, 1.4340048
7: -10.2733612, -8.1275434, -10.2826519, -8.1934061, -1.5155892, 1.5483255
8: -1.9218886, 0.7244959, -1.9271548, 0.7214298, -1.8376632, 1.8475609
9: -8.5035896, -6.9918408, -8.5188007, -6.9964218, -1.2762322, 1.2821808

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445934, upper bound: 0.4431820
time: 5.32 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445934, upper bound: 0.4463885
time: 4.95 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8106360, -3.0571396, -4.8182135, -3.0648367, -1.1007373, 1.1176894
1: -2.8686416, -0.8980811, -2.8852155, -0.8973541, -1.3432155, 1.3464571
2: -4.2140980, -2.3600872, -4.2134666, -2.3791633, -1.4127078, 1.4375124
3: -12.6504002, -9.9024258, -12.6484966, -9.9223213, -1.7912960, 1.7989008
4: -6.0712285, -4.2871799, -6.0557876, -4.2783637, -1.3515015, 1.3258786
5: -2.8980389, -1.0142703, -2.8361478, -1.0166421, -1.3656211, 1.3443179
6: 2.2106287, 3.8323410, 2.2605381, 3.8323970, -1.4679720, 1.4340501
7: -10.2733631, -8.1275425, -10.2826567, -8.1933804, -1.5156264, 1.5483279
8: -1.9218903, 0.7244949, -1.9271562, 0.7214293, -1.8376641, 1.8475618
9: -8.5035915, -6.9918408, -8.5188046, -6.9964256, -1.2762351, 1.2821822

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4477688, upper bound: 0.4431851
time: 4.89 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4477688, upper bound: 0.4431855
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.8278694, -3.0454874, -4.8264956, -3.0634727, -1.1232333, 1.1315746
1: -2.9038630, -0.8953258, -2.8946757, -0.8970212, -1.3616710, 1.3541968
2: -4.2190580, -2.3537941, -4.2149153, -2.3779716, -1.4214213, 1.4458764
3: -12.6531239, -9.8759575, -12.6488686, -9.9133253, -1.8038056, 1.8177121
4: -6.0864539, -4.2580161, -6.0573196, -4.2626247, -1.3728006, 1.3566258
5: -2.9087780, -1.0042229, -2.8368387, -1.0104532, -1.3772011, 1.3570807
6: 2.1995411, 3.8344631, 2.2568297, 3.8324933, -1.4738140, 1.4403260
7: -10.2900696, -8.1258163, -10.2832747, -8.1932316, -1.5244339, 1.5510864
8: -1.9414353, 0.7272563, -1.9330206, 0.7219803, -1.8663907, 1.8628173
9: -8.5248032, -6.9859409, -8.5199604, -6.9936194, -1.2857318, 1.2876074

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443824, upper bound: 0.4434276
time: 5.09 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443824, upper bound: 0.4466313
time: 4.90 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8278694, -3.0454888, -4.8264980, -3.0634696, -1.1232443, 1.1315763
1: -2.9038634, -0.8953245, -2.8946738, -0.8970168, -1.3616726, 1.3542007
2: -4.2190580, -2.3538022, -4.2149153, -2.3779712, -1.4214211, 1.4458771
3: -12.6531239, -9.8759575, -12.6488714, -9.9133005, -1.8038082, 1.8177128
4: -6.0864530, -4.2580137, -6.0573597, -4.2626200, -1.3728023, 1.3566675
5: -2.9087634, -1.0042224, -2.8368411, -1.0104561, -1.3772023, 1.3570824
6: 2.1995411, 3.8344660, 2.2568014, 3.8324981, -1.4738159, 1.4403722
7: -10.2900715, -8.1258154, -10.2832775, -8.1932068, -1.5244703, 1.5510886
8: -1.9414365, 0.7272558, -1.9330208, 0.7219815, -1.8663902, 1.8628192
9: -8.5248051, -6.9859409, -8.5199642, -6.9936218, -1.2857344, 1.2876091

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430
type: A, layer: 1, pos: 6114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4475592, upper bound: 0.4434310
time: 4.78 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4475592, upper bound: 0.4468507
time: 4.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.8100910, -3.0604291, -4.8640304, -3.0645807, -1.0996807, 1.1115746
1: -2.8659062, -0.8982424, -2.8733499, -0.8687603, -1.3476341, 1.3489304
2: -4.2121396, -2.3598800, -4.2183371, -2.3620181, -1.4258773, 1.4404712
3: -12.6485519, -9.9023142, -12.6525564, -9.9052448, -1.7866490, 1.8011599
4: -6.0698786, -4.2894087, -6.0534210, -4.2713747, -1.3519711, 1.3189709
5: -2.8987229, -1.0189874, -2.9135041, -1.0073123, -1.3670821, 1.3444948
6: 2.2121494, 3.8321543, 2.2585521, 3.8468966, -1.4700339, 1.4366720
7: -10.2691240, -8.1269789, -10.2826748, -8.1213026, -1.5233161, 1.5543067
8: -1.9212549, 0.7207525, -1.9720848, 0.7305484, -1.8448076, 1.8652737
9: -8.5016279, -6.9968853, -8.5882683, -6.9830627, -1.2810469, 1.2797732

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4431849, upper bound: 0.4475268
time: 4.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466097, upper bound: 0.4476777
time: 4.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8190403, -3.0501194, -4.8640304, -3.0645807, -1.1100316, 1.1158490
1: -2.8916721, -0.8958335, -2.8733499, -0.8687603, -1.3536580, 1.3512046
2: -4.2156448, -2.3547878, -4.2183371, -2.3620181, -1.4309447, 1.4458776
3: -12.6508923, -9.8848677, -12.6525564, -9.9052448, -1.7893012, 1.8102131
4: -6.0835457, -4.2759848, -6.0534210, -4.2713747, -1.3585992, 1.3317738
5: -2.9087582, -1.0151262, -2.9135041, -1.0073123, -1.3696275, 1.3482389
6: 2.2047949, 3.8341770, 2.2585521, 3.8468966, -1.4740045, 1.4389625
7: -10.2852192, -8.1254215, -10.2826748, -8.1213026, -1.5256817, 1.5557804
8: -1.9349356, 0.7229607, -1.9720848, 0.7305484, -1.8614659, 1.8683124
9: -8.5216799, -6.9937730, -8.5882683, -6.9830627, -1.2874637, 1.2821016

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4431849, upper bound: 0.4477655
time: 4.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466097, upper bound: 0.4479159
time: 4.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.8100910, -3.0604291, -4.8729830, -3.0542111, -1.1107688, 1.1175432
1: -2.8659062, -0.8982424, -2.8991177, -0.8663718, -1.3497415, 1.3562107
2: -4.2121396, -2.3598800, -4.2218571, -2.3569303, -1.4320772, 1.4441473
3: -12.6485519, -9.9023142, -12.6548958, -9.8878107, -1.7957151, 1.8038101
4: -6.0698786, -4.2894087, -6.0671492, -4.2579536, -1.3579581, 1.3340726
5: -2.8987229, -1.0189874, -2.9235363, -1.0034518, -1.3708286, 1.3470318
6: 2.2121494, 3.8321543, 2.2512059, 3.8489110, -1.4713287, 1.4431858
7: -10.2691240, -8.1269789, -10.2987671, -8.1197405, -1.5247815, 1.5566969
8: -1.9212549, 0.7207525, -1.9857705, 0.7327738, -1.8481326, 1.8720896
9: -8.5016279, -6.9968853, -8.6083088, -6.9799376, -1.2833669, 1.2861965

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4431843, upper bound: 0.4475247
time: 5.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466091, upper bound: 0.4476809
time: 5.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8273268, -3.0487797, -4.8812675, -3.0528378, -1.1292408, 1.1314280
1: -2.9011297, -0.8954861, -2.9085758, -0.8660356, -1.3613684, 1.3639735
2: -4.2170992, -2.3535964, -4.2233124, -2.3557348, -1.4405608, 1.4525120
3: -12.6512756, -9.8758469, -12.6552696, -9.8787918, -1.8081393, 1.8226228
4: -6.0850973, -4.2602415, -6.0687342, -4.2422099, -1.3792720, 1.3648782
5: -2.9094467, -1.0089393, -2.9242291, -0.9972644, -1.3824081, 1.3597970
6: 2.2010622, 3.8342776, 2.2474687, 3.8490112, -1.4771702, 1.4495094
7: -10.2858343, -8.1252508, -10.2993803, -8.1195679, -1.5284262, 1.5594592
8: -1.9408045, 0.7235110, -1.9916320, 0.7333224, -1.8768611, 1.8845999
9: -8.5228367, -6.9909868, -8.6094742, -6.9771433, -1.2928681, 1.2916212

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4431843, upper bound: 0.4475255
time: 4.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466091, upper bound: 0.4476754
time: 5.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.8669472, -3.0496976, -4.8656311, -3.0644615, -1.1148605, 1.1197376
1: -2.8798194, -0.8649223, -2.8737514, -0.8668859, -1.3611350, 1.3564687
2: -4.2214775, -2.3350005, -4.2196445, -2.3599916, -1.4238820, 1.4515285
3: -12.6553745, -9.8653450, -12.6529732, -9.9036846, -1.7844386, 1.8098857
4: -6.0819383, -4.2628851, -6.0540910, -4.2663188, -1.3735480, 1.3607528
5: -2.9910927, -1.0056617, -2.9177041, -1.0072756, -1.3788304, 1.3562164
6: 2.2024651, 3.8498626, 2.2583094, 3.8479981, -1.4818380, 1.4479668
7: -10.2859516, -8.0487232, -10.2833376, -8.1176348, -1.5371780, 1.5646536
8: -1.9854169, 0.7322993, -1.9774795, 0.7306542, -1.8834934, 1.8789077
9: -8.5970011, -6.9793034, -8.5934982, -6.9825211, -1.2990370, 1.2984285

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443527, upper bound: 0.4464622
time: 5.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4443527, upper bound: 0.4467010
time: 5.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.8670168, -3.0496860, -4.8753257, -3.0539591, -1.1193991, 1.1267118
1: -2.8798552, -0.8647593, -2.9004061, -0.8644660, -1.3633902, 1.3631904
2: -4.2215023, -2.3349690, -4.2232838, -2.3547561, -1.4314985, 1.4554105
3: -12.6554203, -9.8652563, -12.6553459, -9.8853931, -1.8037114, 1.8127613
4: -6.0819621, -4.2628312, -6.0680008, -4.2514715, -1.3806601, 1.3676758
5: -2.9911084, -1.0055976, -2.9278281, -1.0028660, -1.3836169, 1.3589547
6: 2.2024260, 3.8498719, 2.2505603, 3.8500357, -1.4832571, 1.4573884
7: -10.2859688, -8.0485449, -10.2995100, -8.1160393, -1.5387626, 1.5673497
8: -1.9854910, 0.7323537, -1.9917459, 0.7329693, -1.8870444, 1.8864989
9: -8.5970173, -6.9791517, -8.6136932, -6.9791455, -1.3015823, 1.3053629

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 430

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445933, upper bound: 0.4464656
time: 4.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445933, upper bound: 0.4464632
time: 5.15 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.85 seconds
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443544, upper bound: 0.4431830
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443544, upper bound: 0.4463894
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4475317, upper bound: 0.4431858
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4475317, upper bound: 0.4431820
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4434215
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443535, upper bound: 0.4466274
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4475308, upper bound: 0.4434243
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4475308, upper bound: 0.4468425
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4445934, upper bound: 0.4431820
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4445934, upper bound: 0.4463885
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4477688, upper bound: 0.4431851
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4477688, upper bound: 0.4431855
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443824, upper bound: 0.4434276
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443824, upper bound: 0.4466313
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4475592, upper bound: 0.4434310
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4475592, upper bound: 0.4468507
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4431849, upper bound: 0.4475268
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4466097, upper bound: 0.4476777
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4431849, upper bound: 0.4477655
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4466097, upper bound: 0.4479159
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4431843, upper bound: 0.4475247
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4466091, upper bound: 0.4476809
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4431843, upper bound: 0.4475255
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4466091, upper bound: 0.4476754
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443527, upper bound: 0.4464622
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4443527, upper bound: 0.4467010
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4445933, upper bound: 0.4464656
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.85
Output dim: 6, lower bound: -0.4445933, upper bound: 0.4464632
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.85
Output dim: 6, lower bound: -0.4476800, upper bound: 0.4468589
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.85
Output dim: 6, lower bound: -0.4479242, upper bound: 0.4468659
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.4518237113952637
rel_dist={6: [-0.44793373691976335, 0.4479324567017895]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 2416.14 seconds
