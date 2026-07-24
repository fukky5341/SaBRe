## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.02634580263
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.8640556, -11.6210804, -15.8640556, -11.6210804, -3.7144022, 3.7144022)
1: (-7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716)
2: (-8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809)
3: (-5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.6043229, 2.6043229)
4: (-7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.6161270, 2.6161273)
5: (-6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.6302273, 2.6302273)
6: (-14.4134359, -10.9648418, -14.4134359, -10.9648418, -3.2066069, 3.2066069)
7: (2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.4982886, 2.4982884)
8: (-1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.3115258, 2.3115258)
9: (-8.8183250, -5.7160292, -8.8183250, -5.7160292, -3.0185328, 3.0185328)

## BASE Result
execution time: IAR + LP analysis = 12.71 + 32.24 = 44.95 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3555.05 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.045863628387451
rel_dist={7: [-1.3815082971736095, 1.3815080831477027]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8196511268615723
rel_dist={7: [-1.029880698625119, 1.029879313881131]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.6688427925109863
rel_dist={7: [-0.7506851770062117, 0.7506857888818104]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7442469596862793
rel_dist={7: [-0.897989859838467, 0.8979910480313595]}

## Binary Search Result
Binary search time: 196.65 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3358.40 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841240, upper bound: 1.4709684
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709688, upper bound: 1.4841237
time: 4.15 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.73
Output dim: 7, lower bound: -1.4841240, upper bound: 1.4709684
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.73
Output dim: 7, lower bound: -1.4709688, upper bound: 1.4841237

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9628043, 2.9801846
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5323906, 2.5278549
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.2004457, 2.2019110
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5019727, 2.4958973
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6285667, 2.6183887
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.1051331, 2.0985076
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1627636, 2.1700921
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4725518, 2.4618781

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841229, upper bound: 1.4659992
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791546, upper bound: 1.4709674
time: 3.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9801841, 2.9628043
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5278549, 2.5323906
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.2019110, 2.2004457
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4958978, 2.5019720
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6183891, 2.6285665
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0985074, 2.1051331
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1700921, 2.1627638
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4618778, 2.4725523

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709678, upper bound: 1.4791541
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4659994, upper bound: 1.4841226
time: 3.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.41
Output dim: 7, lower bound: -1.4841229, upper bound: 1.4659992
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.41
Output dim: 7, lower bound: -1.4791546, upper bound: 1.4709674
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.41
Output dim: 7, lower bound: -1.4709678, upper bound: 1.4791541
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.41
Output dim: 7, lower bound: -1.4659994, upper bound: 1.4841226

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9536958, 2.9745436
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5253935, 2.5179777
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.2000303, 2.2010710
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4949532, 2.4919171
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6011372, 2.5989647
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0937982, 2.0811651
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1613355, 2.1679397
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4813576, 2.4626570

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4812106, upper bound: 1.4659872
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841112, upper bound: 1.4630869
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9571633, 2.9710765
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5225139, 2.5208573
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1996059, 2.2014961
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4979916, 2.4888787
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6091423, 2.5909593
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0877905, 2.0871730
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1606116, 2.1686635
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4733310, 2.4706836

Time for backsubstitution: 13.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4762423, upper bound: 1.4709557
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791428, upper bound: 1.4680552
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9710755, 2.9571631
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5208573, 2.5225139
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.2014961, 2.1996057
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4888792, 2.4979916
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5909586, 2.6091423
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0871730, 2.0877905
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1686635, 2.1606116
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4706831, 2.4733312

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4680554, upper bound: 1.4791426
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709560, upper bound: 1.4762420
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9745431, 2.9536960
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5179782, 2.5253935
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.2010708, 2.2000308
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4919176, 2.4949532
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5989637, 2.6011372
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0811648, 2.0937982
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1679397, 2.1613352
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4626570, 2.4813576

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4630871, upper bound: 1.4841109
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4659877, upper bound: 1.4812104
time: 3.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4812106, upper bound: 1.4659872
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4841112, upper bound: 1.4630869
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4762423, upper bound: 1.4709557
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4791428, upper bound: 1.4680552
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4680554, upper bound: 1.4791426
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4709560, upper bound: 1.4762420
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4630871, upper bound: 1.4841109
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.4659877, upper bound: 1.4812104

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9505558, 2.9660487
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5247273, 2.5161610
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1964188, 2.1911545
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4948959, 2.4917588
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5912676, 2.5953550
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0924001, 2.0806503
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1604280, 2.1654553
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4651361, 2.4566936

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4812004, upper bound: 1.4608098
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4760323, upper bound: 1.4659797
time: 4.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9452009, 2.9714031
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5235763, 2.5173130
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1901140, 2.1974595
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4947948, 2.4918590
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5975275, 2.5890946
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0932837, 2.0797672
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1588507, 2.1670322
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4753952, 2.4464355

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841009, upper bound: 1.4579111
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4789328, upper bound: 1.4630793
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9540224, 2.9625816
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5218482, 2.5190401
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1959934, 2.1915796
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4979343, 2.4887204
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5992727, 2.5873499
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0863924, 2.0866582
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1597042, 2.1661792
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4571095, 2.4647200

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4762346, upper bound: 1.4657774
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4710647, upper bound: 1.4709456
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9486685, 2.9679360
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5206966, 2.5201921
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1896887, 2.1978846
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4978333, 2.4888206
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6055326, 2.5810895
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0872760, 2.0857749
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1581273, 2.1677561
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4673681, 2.4544618

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791351, upper bound: 1.4628769
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4739652, upper bound: 1.4680451
time: 3.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9679356, 2.9486682
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5201921, 2.5206966
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1978846, 2.1896892
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4888210, 2.4978333
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5810900, 2.6055329
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0857749, 2.0872757
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1677561, 2.1581273
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4544616, 2.4673686

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4680452, upper bound: 1.4739649
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4628771, upper bound: 1.4791349
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9625816, 2.9540224
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5190401, 2.5218482
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1915798, 2.1959939
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4887199, 2.4979336
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5873499, 2.5992725
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0866580, 2.0863924
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1661792, 2.1597042
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4647198, 2.4571095

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709458, upper bound: 1.4710644
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4657777, upper bound: 1.4762345
time: 3.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9714031, 2.9452009
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5173125, 2.5235763
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1974592, 2.1901143
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4918594, 2.4947948
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5890951, 2.5975277
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0797672, 2.0932837
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1670322, 2.1588509
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4464350, 2.4753950

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4630794, upper bound: 1.4789326
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4579095, upper bound: 1.4841008
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9660492, 2.9505553
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5161605, 2.5247278
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1911545, 2.1964192
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4917583, 2.4948952
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5953550, 2.5912673
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0806503, 2.0924003
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1654553, 2.1604278
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4566936, 2.4651361

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4659800, upper bound: 1.4760320
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4608101, upper bound: 1.4812001
time: 3.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4812004, upper bound: 1.4608098
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4760323, upper bound: 1.4659797
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4841009, upper bound: 1.4579111
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4789328, upper bound: 1.4630793
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4762346, upper bound: 1.4657774
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4710647, upper bound: 1.4709456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4791351, upper bound: 1.4628769
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4739652, upper bound: 1.4680451
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4680452, upper bound: 1.4739649
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4628771, upper bound: 1.4791349
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4709458, upper bound: 1.4710644
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4657777, upper bound: 1.4762345
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4630794, upper bound: 1.4789326
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4579095, upper bound: 1.4841008
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4659800, upper bound: 1.4760320
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.67
Output dim: 7, lower bound: -1.4608101, upper bound: 1.4812001

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9412708, 2.9529448
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5257568, 2.5174928
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1773787, 2.1631775
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5018997, 2.5008278
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5865116, 2.5936635
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0829439, 2.0649230
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1453900, 2.1550252
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4619331, 2.4370506

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4656454, upper bound: 1.4607897
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4811802, upper bound: 1.4452437
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9374514, 2.9567621
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5260596, 2.5171895
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1684418, 2.1721148
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5039644, 2.4987631
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5895739, 2.5905988
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0766726, 2.0711977
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1499963, 2.1504178
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4454927, 2.4534998

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4604879, upper bound: 1.4659601
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4760122, upper bound: 1.4504026
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9359159, 2.9582992
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5246053, 2.5186448
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1710739, 2.1694822
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5017996, 2.5009282
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5927715, 2.5874031
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0838270, 2.0640399
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1438131, 2.1566021
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4721923, 2.4267924

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4685530, upper bound: 1.4578891
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840808, upper bound: 1.4423361
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9320974, 2.9621165
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5249085, 2.5183415
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1621370, 2.1784196
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5038643, 2.4988635
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5958347, 2.5843387
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0775561, 2.0703146
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1484194, 2.1519947
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4557519, 2.4432416

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4633924, upper bound: 1.4630591
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4789127, upper bound: 1.4474950
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9447355, 2.9494777
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5228772, 2.5203719
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1769543, 2.1636026
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5049381, 2.4977894
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5945168, 2.5856564
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0769401, 2.0709307
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1446662, 2.1557477
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4539161, 2.4450769

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4606272, upper bound: 1.4657573
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4762144, upper bound: 1.4502370
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9409189, 2.9532969
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5231805, 2.5200691
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1680169, 2.1725392
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5070028, 2.4957247
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5975809, 2.5825937
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0706654, 2.0772018
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1492739, 2.1511414
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4374666, 2.4615173

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4554796, upper bound: 1.4709256
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4710446, upper bound: 1.4554053
time: 3.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9393816, 2.9548321
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5217261, 2.5215240
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1706495, 2.1699073
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5048380, 2.4978900
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6007767, 2.5793962
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0778232, 2.0700476
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1430893, 2.1573246
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4641747, 2.4348187

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4635438, upper bound: 1.4628565
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791150, upper bound: 1.4473365
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9355640, 2.9586513
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5220294, 2.5212212
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1617122, 2.1788440
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5069027, 2.4958251
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6038418, 2.5763333
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0715485, 2.0763187
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1476970, 2.1527183
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4477253, 2.4512591

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4583879, upper bound: 1.4680252
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4739451, upper bound: 1.4525013
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9586515, 2.9355643
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5212216, 2.5220289
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1788440, 2.1617122
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4958248, 2.5069022
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5763340, 2.6038411
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0763187, 2.0715482
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1527185, 2.1476970
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4512591, 2.4477253

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4525013, upper bound: 1.4739449
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4680251, upper bound: 1.4583877
time: 4.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9548321, 2.9393816
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5215240, 2.5217257
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1699076, 2.1706495
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4978905, 2.5048378
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5793962, 2.6007767
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0700474, 2.0778232
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1573248, 2.1430895
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4348187, 2.4641747

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4473367, upper bound: 1.4791148
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4628569, upper bound: 1.4635435
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9532967, 2.9409184
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5200691, 2.5231805
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1725392, 2.1680169
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4957247, 2.5070026
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5825939, 2.5975809
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0772018, 2.0706651
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1511412, 2.1492739
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4615173, 2.4374666

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4554053, upper bound: 1.4710443
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709256, upper bound: 1.4554794
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9494772, 2.9447360
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5203724, 2.5228772
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1636028, 2.1769543
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4977894, 2.5049381
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5856571, 2.5945163
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0709305, 2.0769401
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1557474, 2.1446662
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4450769, 2.4539161

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4502372, upper bound: 1.4762143
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4657575, upper bound: 1.4606270
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9621162, 2.9320970
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5183420, 2.5249081
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1784196, 2.1621370
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4988632, 2.5038640
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5843391, 2.5958343
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0703144, 2.0775561
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1519947, 2.1484194
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4432416, 2.4557519

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4474949, upper bound: 1.4789122
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4630592, upper bound: 1.4633941
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9582987, 2.9359164
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5186448, 2.5246053
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1694822, 2.1710739
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5009279, 2.5017993
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5874023, 2.5927715
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0640402, 2.0838273
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1566019, 2.1438131
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4267921, 2.4721923

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4423363, upper bound: 1.4840810
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4578893, upper bound: 1.4685547
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9567623, 2.9374514
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5171900, 2.5260596
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1721148, 2.1684420
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4987631, 2.5039644
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5905991, 2.5895739
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0711975, 2.0766728
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1504178, 2.1499963
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4534998, 2.4454930

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4504031, upper bound: 1.4760118
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4659598, upper bound: 1.4604878
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9529448, 2.9412708
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5174928, 2.5257564
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1631775, 2.1773787
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.5008278, 2.5018997
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.5936642, 2.5865111
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0649233, 2.0829442
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1550255, 2.1453900
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4370508, 2.4619331

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4452438, upper bound: 1.4811806
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4607899, upper bound: 1.4656452
time: 4.08 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4656454, upper bound: 1.4607897
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4811802, upper bound: 1.4452437
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4604879, upper bound: 1.4659601
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4760122, upper bound: 1.4504026
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4685530, upper bound: 1.4578891
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4840808, upper bound: 1.4423361
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4633924, upper bound: 1.4630591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4789127, upper bound: 1.4474950
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4606272, upper bound: 1.4657573
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4762144, upper bound: 1.4502370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4554796, upper bound: 1.4709256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4710446, upper bound: 1.4554053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4635438, upper bound: 1.4628565
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4791150, upper bound: 1.4473365
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4583879, upper bound: 1.4680252
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4739451, upper bound: 1.4525013
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4525013, upper bound: 1.4739449
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4680251, upper bound: 1.4583877
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4473367, upper bound: 1.4791148
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4628569, upper bound: 1.4635435
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4554053, upper bound: 1.4710443
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4709256, upper bound: 1.4554794
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4502372, upper bound: 1.4762143
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4657575, upper bound: 1.4606270
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4474949, upper bound: 1.4789122
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4630592, upper bound: 1.4633941
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4423363, upper bound: 1.4840810
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4578893, upper bound: 1.4685547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4504031, upper bound: 1.4760118
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4659598, upper bound: 1.4604878
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4452438, upper bound: 1.4811806
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.72
Output dim: 7, lower bound: -1.4607899, upper bound: 1.4656452

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9185562, 2.9209185
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5815620, 2.5785966
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5253916, 2.5182967
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1456256, 2.1406746
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4805946, 2.4707742
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6012611, 2.6131372
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0417688, 2.0357387
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1443872, 2.1527500
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4663696, 2.4428902

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4656415, upper bound: 1.4571723
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4620462, upper bound: 1.4607860
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9092445, 2.9302304
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5939159, 2.5662417
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5265608, 2.5171280
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1548762, 2.1314247
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4718456, 2.4795227
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6059856, 2.6084135
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0537598, 2.0237479
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1431150, 2.1540225
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4677730, 2.4414871

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4811765, upper bound: 1.4416434
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4775644, upper bound: 1.4452397
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9147377, 2.9247358
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5728168, 2.5873446
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5256948, 2.5179935
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1366897, 2.1496117
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4826593, 2.4687095
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6043243, 2.6100726
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0354974, 2.0420136
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1489935, 2.1481426
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4499297, 2.4593394

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4604840, upper bound: 1.4623442
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4568762, upper bound: 1.4659560
time: 4.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9054260, 2.9340477
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5851727, 2.5749898
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5268636, 2.5168247
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1459394, 2.1403620
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4739113, 2.4774580
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6090469, 2.6053488
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0474889, 2.0300226
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1477213, 2.1494150
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4513326, 2.4579363

Time for backsubstitution: 14.01 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.121267795562744
rel_dist={7: [-1.4841364466020996, 1.4841360647073234]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1530073, upper bound: 1.1455481
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455477, upper bound: 1.1530065
time: 4.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.99
Output dim: 7, lower bound: -1.1530073, upper bound: 1.1455481
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.99
Output dim: 7, lower bound: -1.1455477, upper bound: 1.1530065

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5476499, 2.5575814
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8076048, 2.8105783
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4676247, 2.4694042
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3505268, 2.3479347
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9486628, 1.9495003
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3002272, 2.2967556
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2872524, 2.2814367
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8760810, 1.8722954
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9871893, 1.9913769
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1559620, 2.1498628

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1530065, upper bound: 1.1425924
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1500523, upper bound: 1.1455462
time: 4.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5575814, 2.5476499
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8105783, 2.8076043
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4694042, 2.4676251
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3479347, 2.3505268
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9495001, 1.9486630
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2967558, 2.3002269
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2814364, 2.2872524
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8722954, 1.8760812
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9913769, 1.9871893
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1498628, 2.1559622

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455472, upper bound: 1.1500517
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425933, upper bound: 1.1530058
time: 4.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 7, lower bound: -1.1530065, upper bound: 1.1425924
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 7, lower bound: -1.1500523, upper bound: 1.1455462
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 7, lower bound: -1.1455472, upper bound: 1.1500517
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 7, lower bound: -1.1425933, upper bound: 1.1530058

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5385423, 2.5504541
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7890911, 2.7951527
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4534950, 2.4524446
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3422952, 2.3380580
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9480658, 1.9486604
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2932081, 2.2914732
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2598228, 2.2585816
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8621716, 1.8549528
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9854507, 1.9892247
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1613274, 2.1506417

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506307, upper bound: 1.1425830
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529972, upper bound: 1.1402071
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5405221, 2.5484734
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7921782, 2.7920656
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4506645, 2.4552732
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3406501, 2.3397036
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9478226, 1.9489033
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2949443, 2.2897367
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2643976, 2.2540073
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8587384, 1.8583856
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9850373, 1.9896383
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1567411, 2.1552281

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1476768, upper bound: 1.1455372
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1500430, upper bound: 1.1431620
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5484738, 2.5405226
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7920656, 2.7921791
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4552727, 2.4506650
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3397036, 2.3406501
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9489031, 1.9478230
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2897367, 2.2949443
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2540073, 2.2643976
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8583856, 1.8587387
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9896383, 1.9850371
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1552281, 2.1567411

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1431625, upper bound: 1.1500425
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455378, upper bound: 1.1476764
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5504537, 2.5385413
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7951527, 2.7890911
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4524441, 2.4534941
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3380580, 2.3422956
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9486599, 1.9480660
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2914729, 2.2932081
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2585812, 2.2598233
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8549528, 1.8621716
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9892249, 1.9854507
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1506414, 2.1613278

Time for backsubstitution: 13.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1402080, upper bound: 1.1529965
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425840, upper bound: 1.1506301
time: 4.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1506307, upper bound: 1.1425830
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1529972, upper bound: 1.1402071
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1476768, upper bound: 1.1455372
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1500430, upper bound: 1.1431620
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1431625, upper bound: 1.1500425
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1455378, upper bound: 1.1476764
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1402080, upper bound: 1.1529965
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.66
Output dim: 7, lower bound: -1.1425840, upper bound: 1.1506301

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5331059, 2.5419593
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7818937, 2.7905488
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4487267, 2.4449692
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3411365, 2.3362408
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9417524, 1.9387438
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2931070, 2.2913148
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2499533, 2.2522893
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8607736, 1.8540595
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9838676, 1.9867404
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1451063, 2.1402819

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506245, upper bound: 1.1394195
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1474676, upper bound: 1.1425778
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5300465, 2.5450191
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7844887, 2.7879553
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4460192, 2.4476771
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3404784, 2.3368993
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9381495, 1.9423466
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2930498, 2.2913721
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2535310, 2.2487118
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8612785, 1.8535547
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9829664, 1.9876413
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1509686, 2.1344199

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529910, upper bound: 1.1370364
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1498377, upper bound: 1.1402022
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5350876, 2.5399780
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7849808, 2.7874613
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4458981, 2.4477983
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3394909, 2.3378863
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9415092, 1.9389868
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2948432, 2.2895784
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2545276, 2.2477150
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8573408, 1.8574924
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9834538, 1.9871540
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1405191, 2.1448684

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1476715, upper bound: 1.1423714
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1445141, upper bound: 1.1455306
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5320282, 2.5430379
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7875757, 2.7848682
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4431896, 2.4505062
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3388329, 2.3385448
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9379063, 1.9425895
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2947860, 2.2896359
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2581048, 2.2441375
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8578453, 1.8569880
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9825525, 1.9880550
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1463814, 2.1390066

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1500377, upper bound: 1.1399905
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1468839, upper bound: 1.1431558
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5430374, 2.5320277
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7848682, 2.7875757
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4505062, 2.4431901
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3385444, 2.3388329
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9425898, 1.9379065
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2896361, 2.2947860
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2441373, 2.2581050
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8569880, 1.8578453
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9880552, 1.9825528
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1390066, 2.1463819

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1431563, upper bound: 1.1468832
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1399909, upper bound: 1.1500371
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5399780, 2.5350871
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7874613, 2.7849813
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4477978, 2.4458981
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3378863, 2.3394909
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9389868, 1.9415092
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2895784, 2.2948432
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2477150, 2.2545278
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8574924, 1.8573408
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9871540, 1.9834538
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1448684, 2.1405196

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455316, upper bound: 1.1445134
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1423721, upper bound: 1.1476705
time: 7.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5450191, 2.5300465
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7879553, 2.7844887
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4476776, 2.4460192
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3368993, 2.3404779
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9423466, 1.9381495
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2913718, 2.2930498
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2487121, 2.2535307
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8535547, 1.8612785
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9876413, 1.9829664
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1344199, 2.1509683

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1402027, upper bound: 1.1498371
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1370368, upper bound: 1.1529905
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5419598, 2.5331059
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7905483, 2.7818937
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4449692, 2.4487267
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3362408, 2.3411360
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9387436, 1.9417522
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2913146, 2.2931070
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2522893, 2.2499533
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8540597, 1.8607738
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9867401, 1.9838674
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1402817, 2.1451061

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425786, upper bound: 1.1474668
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1394205, upper bound: 1.1506235
time: 4.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1506245, upper bound: 1.1394195
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1474676, upper bound: 1.1425778
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1529910, upper bound: 1.1370364
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1498377, upper bound: 1.1402022
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1476715, upper bound: 1.1423714
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1445141, upper bound: 1.1455306
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1500377, upper bound: 1.1399905
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1468839, upper bound: 1.1431558
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1431563, upper bound: 1.1468832
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1399909, upper bound: 1.1500371
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1455316, upper bound: 1.1445134
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1423721, upper bound: 1.1476705
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1402027, upper bound: 1.1498371
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1370368, upper bound: 1.1529905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1425786, upper bound: 1.1474668
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.01
Output dim: 7, lower bound: -1.1394205, upper bound: 1.1506235

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5221844, 2.5288553
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7584152, 2.7709842
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4237189, 2.4149647
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3421655, 2.3374429
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9188819, 1.9107668
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3001118, 2.2994990
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2451973, 2.2492843
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8486300, 1.8383322
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9688296, 1.9743354
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1348572, 2.1206388

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1416345, upper bound: 1.1394111
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506129, upper bound: 1.1304324
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5200024, 2.5310369
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7623291, 2.7670712
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4187217, 2.4199638
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3423386, 2.3372698
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9137754, 1.9158738
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3012915, 2.2983193
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2469473, 2.2475331
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8450460, 1.8419178
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9714618, 1.9717026
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1254630, 2.1300385

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1384804, upper bound: 1.1425661
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1474560, upper bound: 1.1335919
time: 7.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5191250, 2.5319147
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7610102, 2.7683902
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4210105, 2.4176726
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3415074, 2.3381014
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9152789, 1.9143696
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3000546, 2.2995563
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2487745, 2.2457070
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8491344, 1.8378274
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9679284, 1.9752364
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1407194, 2.1147771

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1439886, upper bound: 1.1370278
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529794, upper bound: 1.1280735
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5169420, 2.5340967
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7649240, 2.7644773
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4160151, 2.4226713
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3416810, 2.3379283
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9101725, 1.9194765
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3012342, 2.2983766
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2505250, 2.2439556
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8455510, 1.8414130
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9705606, 1.9726036
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1313252, 2.1241765

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1408353, upper bound: 1.1401937
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1498261, upper bound: 1.1312339
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5241652, 2.5268741
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7615032, 2.7678971
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4208922, 2.4177933
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3405204, 2.3390884
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9186392, 1.9110098
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3018479, 2.2977629
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2497716, 2.2447088
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8451986, 1.8417652
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9684162, 1.9747481
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1302762, 2.1252253

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1386789, upper bound: 1.1423600
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1476599, upper bound: 1.1333880
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5219831, 2.5290565
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7654161, 2.7639837
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4158931, 2.4227901
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3406935, 2.3389153
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9135323, 1.9161165
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3030276, 2.2965829
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2515225, 2.2429585
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8416133, 1.8453486
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9710488, 1.9721160
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1208763, 2.1346197

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1355237, upper bound: 1.1455191
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1445024, upper bound: 1.1365447
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5211048, 2.5299339
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7640982, 2.7653036
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4181838, 2.4205017
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3398623, 2.3397470
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9150362, 1.9146125
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3017902, 2.2978203
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2533488, 2.2411315
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8457036, 1.8412604
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9675150, 1.9756494
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1361384, 2.1193635

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1410353, upper bound: 1.1399788
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1500261, upper bound: 1.1310382
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5189238, 2.5321164
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7680101, 2.7613902
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4131846, 2.4254980
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3400354, 2.3395734
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9099293, 1.9197192
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3029699, 2.2966404
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2550998, 2.2393813
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8421183, 1.8448439
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9701481, 1.9730172
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1267385, 2.1287580

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1378815, upper bound: 1.1431438
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1468723, upper bound: 1.1341962
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5321169, 2.5189238
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7613897, 2.7680111
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4254985, 2.4131851
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3395739, 2.3400350
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9197192, 1.9099295
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2966404, 2.3029702
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2393813, 2.2551000
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8448439, 1.8421180
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9730172, 1.9701478
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1287580, 2.1267388

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1341968, upper bound: 1.1468715
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1431448, upper bound: 1.1378817
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5299339, 2.5211048
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7653036, 2.7640982
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4205012, 2.4181843
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3397470, 2.3398619
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9146128, 1.9150364
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2978201, 2.3017905
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2411313, 2.2533488
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8412604, 1.8457036
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9756494, 1.9675150
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1193638, 2.1361384

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1310389, upper bound: 1.1500255
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1399794, upper bound: 1.1410348
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5290565, 2.5219831
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7639837, 2.7654161
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4227901, 2.4158936
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3389153, 2.3406930
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9161167, 1.9135323
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2965832, 2.3030276
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2429585, 2.2515228
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8453484, 1.8416133
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9721160, 1.9710488
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1346197, 2.1208766

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1365456, upper bound: 1.1445018
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1455200, upper bound: 1.1355217
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5268736, 2.5241647
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7678967, 2.7615037
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4177928, 2.4208922
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3390889, 2.3405199
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9110098, 1.9186392
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2977629, 2.3018479
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2447085, 2.2497716
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8417654, 1.8451989
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9747481, 1.9684160
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1252251, 2.1302762

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1333885, upper bound: 1.1476596
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1423605, upper bound: 1.1386780
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5340967, 2.5169425
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7644768, 2.7649245
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4226718, 2.4160142
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3379283, 2.3416805
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9194765, 1.9101725
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2983766, 2.3012340
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2439556, 2.2505248
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8414130, 1.8455510
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9726038, 1.9705606
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1241765, 2.1313252

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1312317, upper bound: 1.1498252
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1401911, upper bound: 1.1408349
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5319147, 2.5191250
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7683897, 2.7610106
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4176726, 2.4210110
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3381019, 2.3415074
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9143696, 1.9152789
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2995563, 2.3000543
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2457070, 2.2487745
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8378277, 1.8491344
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9752364, 1.9679284
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1147771, 2.1407197

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1280712, upper bound: 1.1529784
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1370253, upper bound: 1.1439880
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5310373, 2.5200024
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7670708, 2.7623296
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4199634, 2.4187222
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3372703, 2.3423386
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9158735, 1.9137752
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2983193, 2.3012915
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2475333, 2.2469473
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8419180, 1.8450463
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9717026, 1.9714618
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1300383, 2.1254630

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1335925, upper bound: 1.1474550
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425670, upper bound: 1.1384798
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5288553, 2.5221844
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7709837, 2.7584157
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4149642, 2.4237189
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3374434, 2.3421655
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9107666, 1.9188819
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2994990, 2.3001115
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2492843, 2.2451973
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8383322, 1.8486297
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9743357, 1.9688296
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1206388, 2.1348574

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1304329, upper bound: 1.1506124
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1394089, upper bound: 1.1416368
time: 4.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1416345, upper bound: 1.1394111
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1506129, upper bound: 1.1304324
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1384804, upper bound: 1.1425661
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1474560, upper bound: 1.1335919
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1439886, upper bound: 1.1370278
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1529794, upper bound: 1.1280735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1408353, upper bound: 1.1401937
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1498261, upper bound: 1.1312339
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1386789, upper bound: 1.1423600
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1476599, upper bound: 1.1333880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1355237, upper bound: 1.1455191
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1445024, upper bound: 1.1365447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1410353, upper bound: 1.1399788
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1500261, upper bound: 1.1310382
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1378815, upper bound: 1.1431438
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1468723, upper bound: 1.1341962
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1341968, upper bound: 1.1468715
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1431448, upper bound: 1.1378817
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1310389, upper bound: 1.1500255
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1399794, upper bound: 1.1410348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1365456, upper bound: 1.1445018
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1455200, upper bound: 1.1355217
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1333885, upper bound: 1.1476596
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1423605, upper bound: 1.1386780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1312317, upper bound: 1.1498252
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1401911, upper bound: 1.1408349
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1280712, upper bound: 1.1529784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1370253, upper bound: 1.1439880
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1335925, upper bound: 1.1474550
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1425670, upper bound: 1.1384798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1304329, upper bound: 1.1506124
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.77
Output dim: 7, lower bound: -1.1394089, upper bound: 1.1416368

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4954796, 2.4968290
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7538481, 2.7674351
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3813648, 2.3796706
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3418007, 2.3377461
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8871293, 1.8842998
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2750573, 2.2694454
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2599473, 2.2667336
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8074548, 1.8040090
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9672818, 1.9720602
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1392941, 2.1258771

Time for backsubstitution: 14.19 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8950552940368652
rel_dist={7: [-1.1530177532447787, 1.1530172261927087]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298720, upper bound: 1.0240443
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240456, upper bound: 1.0298703
time: 7.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.60
Output dim: 7, lower bound: -1.0298720, upper bound: 1.0240443
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.60
Output dim: 7, lower bound: -1.0240456, upper bound: 1.0298703

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4092650, 2.4167132
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7556868, 2.7579174
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4035263, 2.4048610
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2899055, 2.2879615
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8647356, 1.8653636
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2329783, 2.2303751
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1734815, 2.1691194
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7997303, 1.7968912
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9286642, 1.9318051
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0504317, 2.0458574

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298712, upper bound: 1.0218519
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276790, upper bound: 1.0240435
time: 4.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4167132, 2.4092650
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7579174, 2.7556868
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4048615, 2.4035263
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2879615, 2.2899055
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8653636, 1.8647356
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2303753, 2.2329783
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1691194, 2.1734812
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7968912, 1.7997305
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9318051, 1.9286647
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0458574, 2.0504322

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240447, upper bound: 1.0276800
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0218530, upper bound: 1.0298726
time: 4.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.39
Output dim: 7, lower bound: -1.0298712, upper bound: 1.0218519
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.39
Output dim: 7, lower bound: -1.0276790, upper bound: 1.0240435
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.39
Output dim: 7, lower bound: -1.0240447, upper bound: 1.0276800
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.39
Output dim: 7, lower bound: -1.0218530, upper bound: 1.0298726

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4001565, 2.4090915
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7371731, 2.7417197
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3886890, 2.3879008
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2812629, 2.2780848
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8640776, 1.8645234
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2259593, 2.2246585
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1460514, 2.1451209
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7849627, 1.7795486
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9268222, 1.9296529
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0546513, 2.0466366

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277077, upper bound: 1.0218437
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298627, upper bound: 1.0196740
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4016423, 2.4076052
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7394886, 2.7394042
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3865662, 2.3900228
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2800288, 2.2793188
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8638954, 1.8647056
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2272620, 2.2233562
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1494827, 2.1416900
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7823882, 1.7821233
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9265122, 1.9299631
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0512114, 2.0500765

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0255144, upper bound: 1.0240353
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276706, upper bound: 1.0218686
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4076047, 2.4016423
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7394037, 2.7394891
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3900223, 2.3865666
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2793188, 2.2800288
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8647051, 1.8638954
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2233562, 2.2272618
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1416898, 2.1494827
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7821231, 1.7823880
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9299631, 1.9265122
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0500765, 2.0512111

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0218665, upper bound: 1.0276698
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0240362, upper bound: 1.0255135
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4090905, 2.4001565
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7417192, 2.7371736
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3879013, 2.3886881
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2780848, 2.2812629
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8645229, 1.8640776
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2246585, 2.2259595
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1451206, 2.1460519
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7795486, 1.7849627
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9296527, 1.9268224
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0466366, 2.0546510

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0196747, upper bound: 1.0298614
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0218445, upper bound: 1.0277068
time: 4.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0277077, upper bound: 1.0218437
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0298627, upper bound: 1.0196740
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0255144, upper bound: 1.0240353
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0276706, upper bound: 1.0218686
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0218665, upper bound: 1.0276698
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0240362, upper bound: 1.0255135
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0196747, upper bound: 1.0298614
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.35
Output dim: 7, lower bound: -1.0218445, upper bound: 1.0277068

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3939562, 2.4005961
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7299757, 2.7364674
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3832436, 2.3804255
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2799392, 2.2762675
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8568630, 1.8546071
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2258444, 2.2245002
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1361823, 2.1379340
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7835650, 1.7785292
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9250140, 1.9271686
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0384293, 2.0348113

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277032, upper bound: 1.0195362
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0254110, upper bound: 1.0218392
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3916616, 2.4028912
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7319221, 2.7345219
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3812132, 2.3824568
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2794456, 2.2767613
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8541613, 1.8573091
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2258010, 2.2245431
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1388650, 2.1352510
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7839437, 1.7781508
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9243383, 1.9278445
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0428262, 2.0304148

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298582, upper bound: 1.0173653
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275589, upper bound: 1.0196701
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3931475, 2.4014053
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7342377, 2.7322063
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3790913, 2.3845782
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2782116, 2.2779953
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8539791, 1.8574913
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2271037, 2.2232409
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1422958, 2.1318202
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7813687, 1.7807255
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9240279, 1.9281545
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0393863, 2.0338547

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276666, upper bound: 1.0195595
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0253627, upper bound: 1.0218613
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4014053, 2.3931475
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7322063, 2.7342377
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3845787, 2.3790913
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2779956, 2.2782116
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8574915, 1.8539791
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2232409, 2.2271035
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1318202, 2.1422958
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7807255, 1.7813687
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9281545, 1.9240279
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0338545, 2.0393863

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0218620, upper bound: 1.0253618
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0195606, upper bound: 1.0276664
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4028912, 2.3916616
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7345219, 2.7319221
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3824568, 2.3812132
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2767615, 2.2794456
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8573093, 1.8541613
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2245431, 2.2258012
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1352510, 2.1388650
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7781510, 1.7839434
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9278445, 1.9243381
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0304146, 2.0428262

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0196708, upper bound: 1.0275581
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0173663, upper bound: 1.0298571
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4005966, 2.3939562
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7364674, 2.7299757
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3804255, 2.3832440
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2762675, 2.2799392
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8546066, 1.8568633
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2245002, 2.2258444
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1379342, 2.1361821
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7785292, 1.7835648
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9271688, 1.9250140
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0348110, 2.0384295

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0218405, upper bound: 1.0254097
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0195371, upper bound: 1.0277021
time: 4.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0277032, upper bound: 1.0195362
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0254110, upper bound: 1.0218392
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0298582, upper bound: 1.0173653
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0275589, upper bound: 1.0196701
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0276666, upper bound: 1.0195595
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0253627, upper bound: 1.0218613
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0218620, upper bound: 1.0253618
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0195606, upper bound: 1.0276664
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0196708, upper bound: 1.0275581
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0173663, upper bound: 1.0298571
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0218405, upper bound: 1.0254097
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.0195371, upper bound: 1.0277021

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3824892, 2.3874922
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7064981, 2.7159238
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3569865, 2.3504210
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2809682, 2.2774265
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8327160, 1.8266299
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2328491, 2.2323895
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1314259, 2.1344912
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7705250, 1.7628019
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9099760, 1.9141054
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0258322, 2.0151682

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0210325, upper bound: 1.0195273
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276948, upper bound: 1.0128537
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3801947, 2.3897867
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7084436, 2.7139792
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3549552, 2.3524523
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2804751, 2.2779202
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8300142, 1.8293321
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2328057, 2.2324324
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1341090, 2.1318083
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7709036, 1.7624233
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9093003, 1.9147811
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0302286, 2.0107718

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0231855, upper bound: 1.0173566
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298497, upper bound: 1.0106959
time: 6.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3785582, 2.3914232
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7113791, 2.7110443
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3512092, 2.3562012
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2806048, 2.2777903
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8261843, 1.8331623
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2336907, 2.2315476
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1354213, 2.1304948
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7682161, 1.7651126
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9112744, 1.9128065
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0231829, 2.0178216

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0208849, upper bound: 1.0196638
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275504, upper bound: 1.0130002
time: 6.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3816795, 2.3883009
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7107592, 2.7116637
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3528361, 2.3545737
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2792411, 2.2791543
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8298321, 1.8295143
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2341080, 2.2311304
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1375394, 2.1283767
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7683306, 1.7649980
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9089899, 1.9150908
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0267930, 2.0142117

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0209844, upper bound: 1.0195510
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276582, upper bound: 1.0128969
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3883009, 2.3816795
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7116632, 2.7107596
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3545737, 2.3528357
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2791543, 2.2792406
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8295145, 1.8298323
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2311306, 2.2341080
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1283765, 2.1375396
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7649980, 1.7683306
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9150906, 1.9089901
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0142117, 2.0267930

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0128957, upper bound: 1.0276579
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0195522, upper bound: 1.0209832
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3914232, 2.3785577
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7110434, 2.7113795
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3562007, 2.3512082
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2777905, 2.2806046
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8331623, 1.8261843
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2315474, 2.2336907
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1304946, 2.1354215
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7651129, 1.7682159
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9128065, 1.9112744
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0178213, 2.0231831

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0130014, upper bound: 1.0275527
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0196623, upper bound: 1.0208837
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3897867, 2.3801947
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7139788, 2.7084441
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3524528, 2.3549557
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2779202, 2.2804747
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8293324, 1.8300142
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2324324, 2.2328057
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1318083, 2.1341090
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7624235, 1.7709036
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9147811, 1.9093003
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0107718, 2.0302289

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0106970, upper bound: 1.0298485
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0173579, upper bound: 1.0231848
time: 6.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3874922, 2.3824892
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7159233, 2.7064981
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3504214, 2.3569870
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2774267, 2.2809682
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8266296, 1.8327162
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2323895, 2.2328489
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1344914, 2.1314259
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7628021, 1.7705252
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9141054, 1.9099760
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0151682, 2.0258322

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0128548, upper bound: 1.0276935
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0195286, upper bound: 1.0210310
time: 5.11 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0210325, upper bound: 1.0195273
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0276948, upper bound: 1.0128537
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0231855, upper bound: 1.0173566
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0298497, upper bound: 1.0106959
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0208849, upper bound: 1.0196638
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0275504, upper bound: 1.0130002
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0209844, upper bound: 1.0195510
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0276582, upper bound: 1.0128969
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0128957, upper bound: 1.0276579
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0195522, upper bound: 1.0209832
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0130014, upper bound: 1.0275527
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0196623, upper bound: 1.0208837
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0106970, upper bound: 1.0298485
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0173579, upper bound: 1.0231848
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0128548, upper bound: 1.0276935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.0195286, upper bound: 1.0210310

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3504620, 2.3594565
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7026949, 2.7113566
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3199282, 2.3080673
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2811041, 2.2770615
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8049273, 1.7948771
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2027955, 2.2060852
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1482005, 2.1492412
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7344892, 1.7216265
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9077010, 1.9123755
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0308700, 2.0196047

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276906, upper bound: 1.0110243
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0258554, upper bound: 1.0128495
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3481674, 2.3617516
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7046404, 2.7094116
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3178973, 2.3100982
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2806110, 2.2775552
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8022256, 1.7975793
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2027521, 2.2061281
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1508832, 2.1465583
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7348678, 1.7212481
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9070253, 1.9130514
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0352664, 2.0152082

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298456, upper bound: 1.0088553
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0280175, upper bound: 1.0106921
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3465309, 2.3633876
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7075758, 2.7064772
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3141494, 2.3138475
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2807407, 2.2774253
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7983956, 1.8014095
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2036371, 2.2052433
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1521959, 2.1452448
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7321799, 1.7239373
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9089994, 1.9110768
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0282207, 2.0222580

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275462, upper bound: 1.0111622
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0257184, upper bound: 1.0129960
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3496532, 2.3602657
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7069559, 2.7070961
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3157773, 2.3122201
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2793770, 2.2787893
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8020434, 1.7977614
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2040544, 2.2048261
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1543140, 2.1431267
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7322943, 1.7238228
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9067149, 1.9133611
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0318308, 2.0186481

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276539, upper bound: 1.0110535
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0258265, upper bound: 1.0128904
time: 7.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3602657, 2.3496532
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7070961, 2.7069569
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3122201, 2.3157773
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2787895, 2.2793765
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7977614, 1.8020437
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2048259, 2.2040544
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1431270, 2.1543140
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7238228, 1.7322941
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9133611, 1.9067149
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0186481, 2.0318308

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0128915, upper bound: 1.0258254
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0110546, upper bound: 1.0276529
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3633881, 2.3465314
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7064772, 2.7075758
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3138475, 2.3141494
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2774258, 2.2807405
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8014092, 1.7983956
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2052436, 2.2036371
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1452451, 2.1521959
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7239377, 1.7321796
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9110770, 1.9089994
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0222583, 2.0282209

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0129973, upper bound: 1.0257207
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0111609, upper bound: 1.0275448
time: 6.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3617516, 2.3481679
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7094116, 2.7046413
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3100982, 2.3178968
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2775555, 2.2806106
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7975793, 1.8022256
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2061281, 2.2027521
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1465583, 2.1508834
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7212484, 1.7348673
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9130511, 1.9070253
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0152082, 2.0352666

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0106928, upper bound: 1.0280170
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0088565, upper bound: 1.0298446
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3594570, 2.3504629
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7113562, 2.7026949
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3080673, 2.3199277
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2770615, 2.2811041
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7948775, 1.8049276
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2060852, 2.2027953
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1492410, 2.1482005
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7216270, 1.7344887
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9123755, 1.9077010
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0196047, 2.0308700

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0128505, upper bound: 1.0258554
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0110253, upper bound: 1.0276893
time: 5.47 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0276906, upper bound: 1.0110243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0258554, upper bound: 1.0128495
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0298456, upper bound: 1.0088553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0280175, upper bound: 1.0106921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0275462, upper bound: 1.0111622
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0257184, upper bound: 1.0129960
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0276539, upper bound: 1.0110535
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0258265, upper bound: 1.0128904
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0128915, upper bound: 1.0258254
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0110546, upper bound: 1.0276529
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0129973, upper bound: 1.0257207
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0111609, upper bound: 1.0275448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0106928, upper bound: 1.0280170
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0088565, upper bound: 1.0298446
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0128505, upper bound: 1.0258554
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 7, lower bound: -1.0110253, upper bound: 1.0276893

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3497734, 2.3583541
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7022877, 2.7107048
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3149543, 2.3049765
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2724257, 2.2716579
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7941480, 1.7775724
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2007432, 2.2048070
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1412163, 2.1448977
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7342811, 1.7212930
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9034386, 1.9097223
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0201316, 2.0129223

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276906, upper bound: 1.0110242
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0255207, upper bound: 1.0110241
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3474789, 2.3606491
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7042332, 2.7087598
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3129234, 2.3070073
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2719321, 2.2721517
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7914462, 1.7802744
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2006998, 2.2048500
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1438994, 2.1422148
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7346597, 1.7209146
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9027629, 1.9103982
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0245285, 2.0085258

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298455, upper bound: 1.0088553
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276767, upper bound: 1.0088555
time: 5.07 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 26.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 26.20
Output dim: 7, lower bound: -1.0276906, upper bound: 1.0110242
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 26.20
Output dim: 7, lower bound: -1.0255207, upper bound: 1.0110241
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 26.20
Output dim: 7, lower bound: -1.0298455, upper bound: 1.0088553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 26.20
Output dim: 7, lower bound: -1.0276767, upper bound: 1.0088555
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0280175, upper bound: 1.0106921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0275462, upper bound: 1.0111622
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0276539, upper bound: 1.0110535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0110546, upper bound: 1.0276529
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0111609, upper bound: 1.0275448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0106928, upper bound: 1.0280170
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0088565, upper bound: 1.0298446
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.20
Output dim: 7, lower bound: -1.0110253, upper bound: 1.0276893
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8196511268615723
rel_dist={7: [-1.029880698625119, 1.029879313881131]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.09 seconds
