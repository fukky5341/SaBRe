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
execution time: IAR + LP analysis = 15.51 + 32.74 = 48.25 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.75 seconds, max iter: 100)

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
Binary search time: 204.24 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3347.50 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4685884, upper bound: 1.4841161
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841164, upper bound: 1.4685882
time: 4.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.06
Output dim: 7, lower bound: -1.4685884, upper bound: 1.4841161
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.06
Output dim: 7, lower bound: -1.4841164, upper bound: 1.4685882

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9997616, 2.9904499
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5153847, 2.5165534
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1647372, 2.1739872
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4583750, 2.4496264
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6597729, 2.6644964
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0800929, 2.0920835
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1869264, 2.1856539
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5029440, 2.5043468

Time for backsubstitution: 15.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4677715, upper bound: 1.4744984
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4587653, upper bound: 1.4832980
time: 5.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9904499, 2.9997621
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5165539, 2.5153847
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1739874, 2.1647372
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4496264, 2.4583750
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6644964, 2.6597726
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0920835, 2.0800927
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1856542, 2.1869261
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5043468, 2.5029438

Time for backsubstitution: 15.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841153, upper bound: 1.4635755
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791470, upper bound: 1.4685873
time: 4.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.4677715, upper bound: 1.4744984
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.4587653, upper bound: 1.4832980
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.4841153, upper bound: 1.4635755
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 7, lower bound: -1.4791470, upper bound: 1.4685873

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -3.0021629, 2.9936595
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5160136, 2.5170279
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1655540, 2.1745305
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4603701, 2.4525585
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6585989, 2.6628399
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0811334, 2.0927787
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1877828, 2.1863036
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5020852, 2.5045922

Time for backsubstitution: 15.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4677613, upper bound: 1.4693730
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4625997, upper bound: 1.4744885
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -3.0029716, 2.9928508
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5158587, 2.5171824
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1652808, 2.1748040
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4613066, 2.4516218
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6581163, 2.6633229
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0807881, 2.0931242
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1875758, 2.1865101
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5031891, 2.5034885

Time for backsubstitution: 15.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4587529, upper bound: 1.4701307
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4456340, upper bound: 1.4832859
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9813433, 2.9941216
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5923085
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5095572, 2.5055094
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1735711, 2.1638961
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4426084, 2.4543953
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6370659, 2.6403470
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0807481, 2.0627494
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1842256, 2.1847739
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5131526, 2.5037234

Time for backsubstitution: 15.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4841115, upper bound: 1.4599791
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4804995, upper bound: 1.4635716
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9848108, 2.9906549
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5972590
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5066776, 2.5083885
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1731458, 2.1643212
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4456468, 2.4513569
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6450710, 2.6323416
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0747404, 2.0687571
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1835017, 2.1854975
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5051265, 2.5117497

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791432, upper bound: 1.4649805
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4755311, upper bound: 1.4685832
time: 4.20 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4677613, upper bound: 1.4693730
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4625997, upper bound: 1.4744885
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4587529, upper bound: 1.4701307
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4456340, upper bound: 1.4832859
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4841115, upper bound: 1.4599791
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4804995, upper bound: 1.4635716
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4791432, upper bound: 1.4649805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.77
Output dim: 7, lower bound: -1.4755311, upper bound: 1.4685832

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9928832, 2.9805598
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5920205, 2.5943756
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5170417, 2.5183589
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1465077, 2.1465473
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4673758, 2.4616280
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6538415, 2.6611452
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0716758, 2.0770466
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1727476, 2.1758761
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4988775, 2.4849348

Time for backsubstitution: 15.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4677490, upper bound: 1.4562416
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4546002, upper bound: 1.4693605
time: 4.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9890628, 2.9843793
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5832725, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5173445, 2.5180557
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1375709, 2.1554844
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4694395, 2.4595633
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6569047, 2.6580825
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0654006, 2.0833216
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1773548, 2.1712687
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4824276, 2.5013843

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4625958, upper bound: 1.4709273
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4589841, upper bound: 1.4744847
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9432983, 2.9505572
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5324988, 2.5292864
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1692381, 2.1802263
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4835978, 2.4678385
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6416626, 2.6366911
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0646534, 2.0703640
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1624107, 2.1686735
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4772339, 2.4668586

Time for backsubstitution: 15.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4587527, upper bound: 1.4652180
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537105, upper bound: 1.4701304
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9606781, 2.9331768
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5279627, 2.5338221
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1707029, 2.1787610
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4775229, 2.4739132
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6314840, 2.6468687
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0580277, 2.0769894
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1697392, 2.1613452
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4665594, 2.4775329

Time for backsubstitution: 12.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426901, upper bound: 1.4832734
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4456221, upper bound: 1.4803732
time: 7.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9812040, 2.9930191
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5917282
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5008788, 2.5044723
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1714916, 2.1465914
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4405556, 2.4541488
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6300812, 2.6395240
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0807080, 2.0624151
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1799636, 2.1842675
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5024147, 2.5024490

Time for backsubstitution: 12.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4811992, upper bound: 1.4599674
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840998, upper bound: 1.4570599
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9802408, 2.9939842
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5873351
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5085206, 2.4968309
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1562662, 2.1618147
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4423618, 2.4523425
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6362419, 2.6333623
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0804138, 2.0627093
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1837187, 2.1805122
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5118785, 2.4929852

Time for backsubstitution: 12.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4804993, upper bound: 1.4635703
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4755869, upper bound: 1.4635714
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9846716, 2.9895525
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5996904, 2.5966792
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.4979997, 2.5073519
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1710668, 2.1470165
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4435940, 2.4511104
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6380863, 2.6315186
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0747004, 2.0684228
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1792402, 2.1849911
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4943881, 2.5104756

Time for backsubstitution: 12.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791355, upper bound: 1.4598007
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4739656, upper bound: 1.4649704
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9837065, 2.9905171
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5922856
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5056415, 2.4997101
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1558414, 2.1622398
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4454002, 2.4493041
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6442471, 2.6253569
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0744061, 2.0687170
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1829948, 2.1812360
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5038519, 2.5010118

Time for backsubstitution: 12.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4755234, upper bound: 1.4634125
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4703520, upper bound: 1.4685732
time: 4.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4677490, upper bound: 1.4562416
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4546002, upper bound: 1.4693605
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4625958, upper bound: 1.4709273
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4589841, upper bound: 1.4744847
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4587527, upper bound: 1.4652180
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4537105, upper bound: 1.4701304
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4426901, upper bound: 1.4832734
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4456221, upper bound: 1.4803732
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4811992, upper bound: 1.4599674
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4840998, upper bound: 1.4570599
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4804993, upper bound: 1.4635703
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4755869, upper bound: 1.4635714
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4791355, upper bound: 1.4598007
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4739656, upper bound: 1.4649704
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4755234, upper bound: 1.4634125
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.68
Output dim: 7, lower bound: -1.4703520, upper bound: 1.4685732

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9332080, 2.9382658
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6003189, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5336819, 2.5304632
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1504660, 2.1519704
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4896660, 2.4778438
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6373878, 2.6345143
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0555425, 2.0542879
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1475840, 2.1580403
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4729218, 2.4483051

Time for backsubstitution: 12.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4677489, upper bound: 1.4512527
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4627928, upper bound: 1.4562427
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9505887, 2.9208853
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5291457, 2.5349994
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1519318, 2.1505051
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4835911, 2.4839184
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6272101, 2.6446922
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0489168, 2.0609133
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1549120, 2.1507120
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4622474, 2.4589794

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4516857, upper bound: 1.4693488
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4545885, upper bound: 1.4664168
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9889259, 2.9832768
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5782990, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5086670, 2.5170197
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1354890, 2.1381793
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4673872, 2.4593172
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6499200, 2.6572602
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0653610, 2.0829873
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1730928, 2.1707618
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4716897, 2.5001099

Time for backsubstitution: 12.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4596807, upper bound: 1.4709150
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4625841, upper bound: 1.4679830
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9879608, 2.9842415
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5826926, 2.5981507
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5163088, 2.5093780
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1202655, 2.1534047
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4691935, 2.4575109
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6560817, 2.6510985
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0650673, 2.0832815
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1768484, 2.1670067
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4811535, 2.4906464

Time for backsubstitution: 12.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4560718, upper bound: 1.4744728
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4589723, upper bound: 1.4715407
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9432983, 2.9505575
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5324945, 2.5292816
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1692348, 2.1802216
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4835939, 2.4678361
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6416540, 2.6366849
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0646517, 2.0703619
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1624117, 2.1686752
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4772301, 2.4668539

Time for backsubstitution: 12.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4587426, upper bound: 1.4600923
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4536271, upper bound: 1.4652057
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9432983, 2.9505577
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5324941, 2.5292821
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1692328, 2.1802232
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4835958, 2.4678349
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6416559, 2.6366823
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0646508, 2.0703623
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1624126, 2.1686745
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4772286, 2.4668553

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4537093, upper bound: 1.4651611
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4536511, upper bound: 1.4701312
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9575377, 2.9246814
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5272975, 2.5320044
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1670914, 2.1688447
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4774647, 2.4737537
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6216145, 2.6432595
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0566301, 2.0764747
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1688313, 2.1588604
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4503379, 2.4715703

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4426900, upper bound: 1.4783633
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4376478, upper bound: 1.4832738
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9521828, 2.9300358
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5261455, 2.5331564
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1607866, 2.1751497
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4773636, 2.4738541
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6278753, 2.6369991
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0575132, 2.0755916
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1672544, 2.1604373
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4605961, 2.4613113

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4456181, upper bound: 1.4767577
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4420623, upper bound: 1.4803698
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9780636, 2.9845238
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5842538
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5002136, 2.5026546
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1678801, 2.1366751
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4404979, 2.4539907
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6202087, 2.6359127
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0793104, 2.0619011
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1790557, 2.1817825
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4861927, 2.4964852

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4811889, upper bound: 1.4548002
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4760208, upper bound: 1.4599603
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9727097, 2.9898777
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5971656, 2.5889926
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.4990616, 2.5038066
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1615753, 2.1429801
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4403977, 2.4540911
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6264696, 2.6296525
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0801935, 2.0610178
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1774783, 2.1833591
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4964509, 2.4862270

Time for backsubstitution: 12.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4840873, upper bound: 1.4439034
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4709321, upper bound: 1.4570474
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9802408, 2.9939842
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5873313
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5085168, 2.4968257
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1562676, 2.1618147
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4423623, 2.4523439
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6362343, 2.6333561
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0804143, 2.0627091
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1837177, 2.1805120
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5118833, 2.4929883

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4804891, upper bound: 1.4584079
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4753195, upper bound: 1.4635616
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9802408, 2.9939842
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6010809, 2.5873322
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5085154, 2.4968262
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1562662, 2.1618161
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4423633, 2.4523430
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6362362, 2.6333537
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0804133, 2.0627098
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1837168, 2.1805110
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5118818, 2.4929900

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4755746, upper bound: 1.4584081
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4704595, upper bound: 1.4635637
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9753857, 2.9764485
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5784330, 2.5666742
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.4990287, 2.5086834
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1520262, 2.1190388
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4505987, 2.4601800
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6333284, 2.6298246
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0652475, 2.0526953
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1642017, 2.1745591
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4911942, 2.4908321

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791354, upper bound: 1.4549198
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4791334, upper bound: 1.4598006
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9715681, 2.9802680
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5696850, 2.5754185
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.4993320, 2.5083807
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1430893, 2.1279757
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4526634, 2.4581153
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6363935, 2.6267619
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0589728, 2.0589664
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1688089, 2.1699529
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4747453, 2.5072722

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4739531, upper bound: 1.4518026
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4607979, upper bound: 1.4649581
time: 3.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9744215, 2.9774132
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5828266, 2.5622807
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5066705, 2.5010421
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1368008, 2.1342623
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4524050, 2.4583738
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6394911, 2.6236629
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0649533, 2.0529895
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1679564, 2.1708040
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.5006580, 2.4813683

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4755234, upper bound: 1.4585162
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4755213, upper bound: 1.4634123
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9706039, 2.9812326
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5740786, 2.5710249
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5069733, 2.5007393
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1278639, 2.1431990
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4544697, 2.4563091
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6425552, 2.6206002
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0586782, 2.0592606
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1725645, 2.1661978
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4842086, 2.4978085

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4695340, upper bound: 1.4587500
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4607116, upper bound: 1.4677562
time: 4.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4677489, upper bound: 1.4512527
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4627928, upper bound: 1.4562427
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4516857, upper bound: 1.4693488
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4545885, upper bound: 1.4664168
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4596807, upper bound: 1.4709150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4625841, upper bound: 1.4679830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4560718, upper bound: 1.4744728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4589723, upper bound: 1.4715407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4587426, upper bound: 1.4600923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4536271, upper bound: 1.4652057
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4537093, upper bound: 1.4651611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4536511, upper bound: 1.4701312
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4426900, upper bound: 1.4783633
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4376478, upper bound: 1.4832738
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4456181, upper bound: 1.4767577
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4420623, upper bound: 1.4803698
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4811889, upper bound: 1.4548002
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4760208, upper bound: 1.4599603
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4840873, upper bound: 1.4439034
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4709321, upper bound: 1.4570474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4804891, upper bound: 1.4584079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4753195, upper bound: 1.4635616
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4755746, upper bound: 1.4584081
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4704595, upper bound: 1.4635637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4791354, upper bound: 1.4549198
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4791334, upper bound: 1.4598006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4739531, upper bound: 1.4518026
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4607979, upper bound: 1.4649581
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4755234, upper bound: 1.4585162
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4755213, upper bound: 1.4634123
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4695340, upper bound: 1.4587500
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.21
Output dim: 7, lower bound: -1.4607116, upper bound: 1.4677562

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9332075, 2.9382653
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6003160, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5336776, 2.5304585
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1504622, 2.1519649
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4896631, 2.4778426
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6373787, 2.6345072
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0555406, 2.0542855
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1475849, 2.1580427
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4729180, 2.4482996

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4648293, upper bound: 1.4512410
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4677371, upper bound: 1.4483090
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9332075, 2.9382656
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6003151, 2.6010809
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5336771, 2.5304585
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1504607, 2.1519666
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4896641, 2.4778414
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6373806, 2.6345043
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0555401, 2.0542862
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1475859, 2.1580417
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4729166, 2.4483011

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4627889, upper bound: 1.4526807
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4591945, upper bound: 1.4562372
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9474487, 2.9123902
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.6006956, 2.5951986
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5284805, 2.5331821
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1483192, 2.1405888
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4835320, 2.4837592
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6173406, 2.6410825
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0475197, 2.0603988
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1540036, 2.1482270
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4460263, 2.4530168

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4516818, upper bound: 1.4657874
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4480737, upper bound: 1.4693446
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.9420929, 2.9177451
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8157716, 2.8157716
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.5959578, 2.5999374
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.5273285, 2.5343332
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -2.1420145, 2.1468935
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.4834318, 2.4838595
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.6236014, 2.6348221
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -2.0484028, 2.0595157
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.1524267, 2.1498039
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.4562840, 2.4427576

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4545873, upper bound: 1.4613165
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4495852, upper bound: 1.4664155
time: 4.27 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4648293, upper bound: 1.4512410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4677371, upper bound: 1.4483090
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4627889, upper bound: 1.4526807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4591945, upper bound: 1.4562372
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4516818, upper bound: 1.4657874
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4480737, upper bound: 1.4693446
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4545873, upper bound: 1.4613165
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.09
Output dim: 7, lower bound: -1.4495852, upper bound: 1.4664155
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4596807, upper bound: 1.4709150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4625841, upper bound: 1.4679830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4560718, upper bound: 1.4744728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4589723, upper bound: 1.4715407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4587426, upper bound: 1.4600923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4536271, upper bound: 1.4652057
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4537093, upper bound: 1.4651611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4536511, upper bound: 1.4701312
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4426900, upper bound: 1.4783633
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4376478, upper bound: 1.4832738
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4456181, upper bound: 1.4767577
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4420623, upper bound: 1.4803698
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4811889, upper bound: 1.4548002
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4760208, upper bound: 1.4599603
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4840873, upper bound: 1.4439034
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4709321, upper bound: 1.4570474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4804891, upper bound: 1.4584079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4753195, upper bound: 1.4635616
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4755746, upper bound: 1.4584081
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4704595, upper bound: 1.4635637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4791354, upper bound: 1.4549198
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4791334, upper bound: 1.4598006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4739531, upper bound: 1.4518026
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4607979, upper bound: 1.4649581
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4755234, upper bound: 1.4585162
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4755213, upper bound: 1.4634123
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4695340, upper bound: 1.4587500
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.09
Output dim: 7, lower bound: -1.4607116, upper bound: 1.4677562
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.121267795562744
rel_dist={7: [-1.4841364466020996, 1.4841360647073234]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529471, upper bound: 1.1457259
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1457267, upper bound: 1.1529466
time: 5.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.07
Output dim: 7, lower bound: -1.1529471, upper bound: 1.1457259
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.07
Output dim: 7, lower bound: -1.1457267, upper bound: 1.1529466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.6097240, 2.6101856
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7949266, 2.7951612
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4633274, 2.4626122
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3363914, 2.3363030
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9455242, 1.9453678
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2829370, 2.2834718
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3066888, 2.3064132
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8960953, 1.8958981
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0131221, 2.0130041
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1856327, 2.1862636

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529363, upper bound: 1.1382497
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1454554, upper bound: 1.1457144
time: 5.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.6101856, 2.6097236
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7951603, 2.7949271
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4626122, 2.4633274
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3363032, 2.3363917
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9453678, 1.9455240
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2834721, 2.2829368
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3064127, 2.3066890
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8958979, 1.8960953
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0130038, 2.0131218
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1862636, 2.1856327

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1367212, upper bound: 1.1529360
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1457151, upper bound: 1.1439590
time: 5.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -1.1529363, upper bound: 1.1382497
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -1.1454554, upper bound: 1.1457144
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -1.1367212, upper bound: 1.1529360
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -1.1457151, upper bound: 1.1439590

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5500484, 2.5604429
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8088198, 2.8120275
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4716253, 2.4726896
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3510885, 2.3484077
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9494810, 1.9501619
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3026237, 2.2996879
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2858739, 2.2797818
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8771214, 1.8731384
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9879565, 1.9920261
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1551032, 2.1496344

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529362, upper bound: 1.1354288
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1501250, upper bound: 1.1382496
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5599818, 2.5505109
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8117933, 2.8090534
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4734049, 2.4709101
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3484964, 2.3509998
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9503183, 1.9493246
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2991524, 2.3031592
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2800574, 2.2855978
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8733358, 1.8769243
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9921441, 1.9878387
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1490035, 2.1557341

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1364778, upper bound: 1.1457037
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1454437, upper bound: 1.1367089
time: 5.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5834799, 2.5776958
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7905941, 2.7913785
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4202576, 2.4280329
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3359389, 2.3366957
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9136152, 1.9190571
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2584157, 2.2528815
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3211641, 2.3241394
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8547235, 1.8617728
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0114560, 2.0108471
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1907005, 2.1908717

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1367198, upper bound: 1.1499813
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1337605, upper bound: 1.1529343
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5781584, 2.5830173
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7916126, 2.7903595
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4273176, 2.4209728
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3366070, 2.3360271
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9189010, 1.9137714
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2534165, 2.2578807
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3238635, 2.3214400
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8615756, 1.8549209
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0107293, 2.0115740
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1915021, 2.1900702

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1457084, upper bound: 1.1407960
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425549, upper bound: 1.1439552
time: 5.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1529362, upper bound: 1.1354288
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1501250, upper bound: 1.1382496
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1364778, upper bound: 1.1457037
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1454437, upper bound: 1.1367089
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1367198, upper bound: 1.1499813
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1337605, upper bound: 1.1529343
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1457084, upper bound: 1.1407960
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 7, lower bound: -1.1425549, upper bound: 1.1439552

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5500498, 2.5604424
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8088169, 2.8120260
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4716225, 2.4726863
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3510847, 2.3484035
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9494762, 1.9501565
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3026195, 2.2996840
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2858639, 2.2797737
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8771195, 1.8731363
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9879580, 1.9920282
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1550994, 2.1496296

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529313, upper bound: 1.1331193
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506481, upper bound: 1.1354241
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5500498, 2.5604424
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8088179, 2.8120255
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4716225, 2.4726868
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3510842, 2.3484039
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9494753, 1.9501574
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3026199, 2.2996833
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2858653, 2.2797720
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8771195, 1.8731365
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9879589, 1.9920278
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1550984, 2.1496303

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1411617, upper bound: 1.1382381
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1501134, upper bound: 1.1292647
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5332756, 2.5184841
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8072252, 2.8055043
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4310503, 2.4356155
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3481312, 2.3513031
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9185658, 1.9228580
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2740974, 2.2731047
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2948084, 2.3030474
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8321609, 1.8426013
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9905968, 1.9855642
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1534405, 2.1609721

Time for backsubstitution: 13.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1364728, upper bound: 1.1434225
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1341822, upper bound: 1.1456976
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5279541, 2.5238056
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8082438, 2.8044853
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4381104, 2.4285560
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3487992, 2.3506346
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9238515, 1.9175723
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2690983, 2.2781036
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2975073, 2.3003483
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8390126, 1.8357494
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9898696, 1.9862912
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1542420, 2.1601706

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1454378, upper bound: 1.1335514
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1422789, upper bound: 1.1367035
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5743723, 2.5705690
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7720795, 2.7759519
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4061260, 2.4110727
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3277082, 2.3268189
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9130177, 1.9182169
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2513976, 2.2475996
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2937346, 2.3012843
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8408122, 1.8444288
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0097160, 2.0086939
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1960664, 2.1916509

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1343593, upper bound: 1.1499712
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1367105, upper bound: 1.1475899
time: 4.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5763521, 2.5685878
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7751675, 2.7728648
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4032974, 2.4139018
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3260627, 2.3284640
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9127750, 1.9184599
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2531343, 2.2458634
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2983088, 2.2967098
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8373795, 1.8478618
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0093031, 2.0091074
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1914797, 2.1962373

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1337498, upper bound: 1.1454429
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1263099, upper bound: 1.1529232
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5672412, 2.5699177
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7681417, 2.7708015
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4023204, 2.3909760
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3376350, 2.3372285
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8960242, 1.8857880
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2604218, 2.2660654
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3191061, 2.3184328
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8494287, 1.8391888
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9956942, 1.9991722
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1812449, 2.1704125

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1457077, upper bound: 1.1378420
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1427544, upper bound: 1.1407954
time: 5.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5650592, 2.5721002
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7720547, 2.7668886
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3973212, 2.3959746
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3378086, 2.3370554
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8909173, 1.8908951
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2616014, 2.2648857
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3208561, 2.3166826
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8458428, 1.8427744
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9983273, 1.9965391
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1718450, 2.1798122

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425540, upper bound: 1.1411704
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1398095, upper bound: 1.1439525
time: 5.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1529313, upper bound: 1.1331193
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1506481, upper bound: 1.1354241
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1411617, upper bound: 1.1382381
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1501134, upper bound: 1.1292647
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1364728, upper bound: 1.1434225
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1341822, upper bound: 1.1456976
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1454378, upper bound: 1.1335514
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1422789, upper bound: 1.1367035
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1343593, upper bound: 1.1499712
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1367105, upper bound: 1.1475899
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1337498, upper bound: 1.1454429
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1263099, upper bound: 1.1529232
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1457077, upper bound: 1.1378420
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1427544, upper bound: 1.1407954
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1425540, upper bound: 1.1411704
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.29
Output dim: 7, lower bound: -1.1398095, upper bound: 1.1439525

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5494976, 2.5593405
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8084908, 2.8113728
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4666481, 2.4702225
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3424063, 2.3440924
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9408703, 1.9328518
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3005672, 2.2986639
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2788792, 2.2763100
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8769543, 1.8728025
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9836960, 1.9899113
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1443605, 2.1442988

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1439431, upper bound: 1.1331099
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1529196, upper bound: 1.1241762
time: 5.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5489464, 2.5598917
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8081646, 2.8116989
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4691591, 2.4677114
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3467727, 2.3397255
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9321718, 1.9415510
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.3015995, 2.2976317
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2824001, 2.2727890
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8767860, 1.8729706
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9858418, 1.9877656
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1497688, 2.1388910

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1416643, upper bound: 1.1354126
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1506365, upper bound: 1.1264802
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5233440, 2.5284166
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8042488, 2.8084755
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4292669, 2.4373918
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3507185, 2.3487062
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9177237, 1.9236915
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2775655, 2.2696297
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3006167, 2.2972229
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8359444, 1.8388133
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9864106, 1.9897525
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1595349, 2.1548686

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1410266, upper bound: 1.1352848
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1409935, upper bound: 1.1382374
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5180225, 2.5337377
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8052673, 2.8074565
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4363270, 2.4303317
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3513865, 2.3480387
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9230094, 1.9184058
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2725663, 2.2746289
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3033166, 2.2945235
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8427966, 1.8319614
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9856834, 1.9904797
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1603370, 2.1540668

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1500027, upper bound: 1.1263101
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1499696, upper bound: 1.1292644
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5327244, 2.5173821
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8069000, 2.8048520
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4260769, 2.4331527
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3394537, 2.3469915
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9099593, 1.9055529
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2720456, 2.2720847
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2878237, 2.2995842
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8319950, 1.8422675
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9863329, 1.9834466
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1427021, 2.1556420

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1364668, upper bound: 1.1402593
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1333071, upper bound: 1.1434164
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5321732, 2.5179334
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8065739, 2.8051782
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4285879, 2.4306426
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3438201, 2.3426251
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9012609, 1.9142530
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2730775, 2.2710528
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2913446, 2.2960632
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8318272, 1.8424356
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9884791, 1.9813008
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1481099, 2.1502342

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1317840, upper bound: 1.1456882
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1341728, upper bound: 1.1433176
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5170360, 2.5107055
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7847748, 2.7849288
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4131126, 2.3985591
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3498278, 2.3518367
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9009762, 1.8895898
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2761021, 2.2862875
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2927513, 2.2973418
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8268676, 1.8200185
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9748354, 1.9738901
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1439843, 2.1405132

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1454328, upper bound: 1.1312829
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1431391, upper bound: 1.1335460
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5148530, 2.5128875
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7886868, 2.7810149
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4081135, 2.4035583
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3500009, 2.3516636
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8958693, 1.8946967
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2772822, 2.2851076
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2945013, 2.2955918
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8232822, 1.8236041
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9774685, 1.9712570
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1345849, 2.1499128

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1398872, upper bound: 1.1366941
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1422696, upper bound: 1.1343446
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5689354, 2.5620737
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7648821, 2.7713494
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4013600, 2.4035974
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3265481, 2.3250008
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9067044, 1.9083004
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2512960, 2.2474408
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2838626, 2.2949896
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8394151, 1.8435359
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0081320, 2.0062091
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1798444, 2.1812906

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1343586, upper bound: 1.1499703
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1314271, upper bound: 1.1499708
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5658760, 2.5651331
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7674770, 2.7687550
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3986516, 2.4063058
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3258896, 2.3256593
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9031014, 1.9119031
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2512388, 2.2474980
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2874398, 2.2914124
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8399196, 1.8430309
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -2.0072317, 2.0071101
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1857061, 2.1754286

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1367055, upper bound: 1.1452940
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1344471, upper bound: 1.1475846
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5166788, 2.5188465
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7890587, 2.7897301
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4115934, 2.4239774
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3407588, 2.3405681
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9167323, 1.9232543
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2728219, 2.2620802
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2774930, 2.2700779
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8184071, 1.8251028
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9841371, 1.9881294
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1609497, 2.1596074

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1337448, upper bound: 1.1431438
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1314918, upper bound: 1.1454378
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5266104, 2.5089149
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7920322, 2.7867565
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4133730, 2.4221983
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3381667, 2.3431602
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9175696, 1.9224169
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2693505, 2.2655513
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2716770, 2.2758937
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8146210, 1.8288887
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9883246, 1.9839418
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1548500, 2.1657071

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1263099, upper bound: 1.1500019
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1263092, upper bound: 1.1529232
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5581293, 2.5627871
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7496204, 2.7553682
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3881793, 2.3740087
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3294053, 2.3273537
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8954315, 1.8849528
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2534037, 2.2607834
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2916770, 2.2955794
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8355200, 1.8218491
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9939528, 1.9970174
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1866183, 2.1712050

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1433278, upper bound: 1.1378327
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1456983, upper bound: 1.1354550
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5601101, 2.5608053
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7527084, 2.7522817
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3853526, 2.3768373
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3277597, 2.3289988
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8951893, 1.8851957
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2551398, 2.2590473
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2962513, 2.2910039
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8320892, 1.8252819
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9935393, 1.9974301
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1820374, 2.1757917

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1427489, upper bound: 1.1385105
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1404740, upper bound: 1.1407901
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5650578, 2.5720997
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7720518, 2.7668862
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3973174, 2.3959708
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3378034, 2.3370502
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8909135, 1.8908901
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2615981, 2.2648830
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3208480, 2.3166761
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8458428, 1.8427734
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9983287, 1.9965417
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1718402, 2.1798062

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1401712, upper bound: 1.1411631
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1425447, upper bound: 1.1387905
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5650578, 2.5720997
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7720518, 2.7668862
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3973174, 2.3959718
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3378034, 2.3370507
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8909125, 1.8908911
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2615991, 2.2648826
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.3208494, 2.3166745
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8458424, 1.8427737
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9983296, 1.9965413
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1718392, 2.1798072

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1374522, upper bound: 1.1439436
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1398006, upper bound: 1.1415679
time: 4.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1439431, upper bound: 1.1331099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1529196, upper bound: 1.1241762
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1416643, upper bound: 1.1354126
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1506365, upper bound: 1.1264802
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1410266, upper bound: 1.1352848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1409935, upper bound: 1.1382374
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1500027, upper bound: 1.1263101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1499696, upper bound: 1.1292644
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1364668, upper bound: 1.1402593
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1333071, upper bound: 1.1434164
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1317840, upper bound: 1.1456882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1341728, upper bound: 1.1433176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1454328, upper bound: 1.1312829
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1431391, upper bound: 1.1335460
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1398872, upper bound: 1.1366941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1422696, upper bound: 1.1343446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1343586, upper bound: 1.1499703
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1314271, upper bound: 1.1499708
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1367055, upper bound: 1.1452940
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1344471, upper bound: 1.1475846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1337448, upper bound: 1.1431438
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1314918, upper bound: 1.1454378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1263099, upper bound: 1.1500019
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1263092, upper bound: 1.1529232
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1433278, upper bound: 1.1378327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1456983, upper bound: 1.1354550
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1427489, upper bound: 1.1385105
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1404740, upper bound: 1.1407901
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1401712, upper bound: 1.1411631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1425447, upper bound: 1.1387905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1374522, upper bound: 1.1439436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.79
Output dim: 7, lower bound: -1.1398006, upper bound: 1.1415679

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.5227923, 2.5273137
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.8039227, 2.8078237
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.4242945, 2.4349289
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.3420405, 2.3443942
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.9091182, 1.9063849
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2755132, 2.2686107
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.2936311, 2.2937613
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8357790, 1.8384790
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9821482, 1.9876368
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.1487970, 2.1495373

Time for backsubstitution: 14.00 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8950552940368652
rel_dist={7: [-1.1530177532447787, 1.1530172261927087]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277184, upper bound: 1.0298702
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298722, upper bound: 1.0277190
time: 4.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 7, lower bound: -1.0277184, upper bound: 1.0298702
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 7, lower bound: -1.0298722, upper bound: 1.0277190

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4627361, 2.4604416
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7345963, 2.7365427
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3897848, 2.3877535
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2745323, 2.2740383
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8535652, 1.8508632
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2140422, 2.2139997
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1858797, 2.1885626
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8182535, 1.8186319
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9520211, 1.9513454
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0662646, 2.0706608

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277142, upper bound: 1.0280385
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0258777, upper bound: 1.0298661
time: 4.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4604416, 2.4627361
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7365427, 2.7345963
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3877535, 2.3897843
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2740383, 2.2745323
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8508635, 1.8535652
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2139993, 2.2140427
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1885629, 2.1858797
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8186316, 1.8182535
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9513450, 1.9520211
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0706611, 2.0662644

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298713, upper bound: 1.0255218
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276792, upper bound: 1.0277155
time: 4.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.51
Output dim: 7, lower bound: -1.0277142, upper bound: 1.0280385
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.51
Output dim: 7, lower bound: -1.0258777, upper bound: 1.0298661
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.51
Output dim: 7, lower bound: -1.0298713, upper bound: 1.0255218
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.51
Output dim: 7, lower bound: -1.0276792, upper bound: 1.0277155

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4620485, 2.4593396
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7341909, 2.7358918
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3848104, 2.3846631
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2658548, 2.2686360
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8427849, 1.8335586
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2119908, 2.2127223
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1788955, 2.1842194
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8180461, 1.8182981
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9477587, 1.9486921
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0555263, 2.0639789

Time for backsubstitution: 13.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0210419, upper bound: 1.0280328
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277058, upper bound: 1.0213659
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4616346, 2.4597535
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7339458, 2.7361364
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3866949, 2.3827801
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2691298, 2.2653611
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8362603, 1.8400829
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2127652, 2.2119482
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1815362, 2.1815784
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8179202, 1.8184242
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9493680, 1.9470828
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0595822, 2.0599229

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0258691, upper bound: 1.0240314
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0200213, upper bound: 1.0298581
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4513359, 2.4551158
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7180309, 2.7184000
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3729143, 2.3728237
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2653966, 2.2646563
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8502040, 1.8527238
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2069817, 2.2083266
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1611328, 2.1618810
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8038640, 1.8009105
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9495025, 1.9498682
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0748796, 2.0670435

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298627, upper bound: 1.0196740
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0240362, upper bound: 1.0255135
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4528217, 2.4536295
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7203465, 2.7160845
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3707924, 2.3749452
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2641625, 2.2658904
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8500218, 1.8529060
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2082834, 2.2070243
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1645641, 2.1584501
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.8012891, 1.8034852
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9491925, 1.9501786
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0714397, 2.0704834

Time for backsubstitution: 13.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276706, upper bound: 1.0218686
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0218445, upper bound: 1.0277068
time: 4.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0210419, upper bound: 1.0280328
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0277058, upper bound: 1.0213659
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0258691, upper bound: 1.0240314
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0200213, upper bound: 1.0298581
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0298627, upper bound: 1.0196740
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0240362, upper bound: 1.0255135
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0276706, upper bound: 1.0218686
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.14
Output dim: 7, lower bound: -1.0218445, upper bound: 1.0277068

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4340119, 2.4273129
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7296228, 2.7320886
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3424582, 2.3476048
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2654896, 2.2687714
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8110323, 1.8057702
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1856852, 2.1826673
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1936464, 2.2009950
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7768707, 1.7822621
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9460292, 1.9464173
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0599637, 2.0690174

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0209798, upper bound: 1.0225037
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0155123, upper bound: 1.0279712
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4300208, 2.4313035
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7303877, 2.7313242
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3477530, 2.3423095
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2659903, 2.2682707
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8149977, 1.8018060
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1819363, 2.1864166
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1956711, 2.1989703
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7820096, 1.7771230
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9454837, 1.9469624
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0605650, 2.0684161

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6192

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276971, upper bound: 1.0155311
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0218547, upper bound: 1.0213601
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4094110, 2.4000807
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7500677, 2.7500281
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3963280, 2.3910789
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2812347, 2.2794099
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8408446, 1.8440392
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2289810, 2.2307673
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1549067, 2.1593108
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7951598, 1.7985036
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9273429, 1.9219172
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0229535, 2.0278692

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0200166, upper bound: 1.0275542
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0177150, upper bound: 1.0298536
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0231900, upper bound: 1.0196684
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298542, upper bound: 1.0130075
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 13.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276706, upper bound: 1.0197011
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276700, upper bound: 1.0218657
time: 4.78 seconds

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

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0151627, upper bound: 1.0276982
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0218360, upper bound: 1.0210357
time: 6.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0209798, upper bound: 1.0225037
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0155123, upper bound: 1.0279712
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0276971, upper bound: 1.0155311
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0218547, upper bound: 1.0213601
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0200166, upper bound: 1.0275542
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0177150, upper bound: 1.0298536
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0231900, upper bound: 1.0196684
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0298542, upper bound: 1.0130075
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0276706, upper bound: 1.0197011
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0276700, upper bound: 1.0218657
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0151627, upper bound: 1.0276982
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.88
Output dim: 7, lower bound: -1.0218360, upper bound: 1.0210357

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4367576, 2.4297128
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7310123, 2.7333021
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3459210, 2.3516045
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2659626, 2.2693110
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8117323, 1.8065872
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1886177, 2.1851981
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1919909, 2.1995459
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7777634, 1.7833023
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9466767, 1.9471536
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0595779, 2.0681584

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6192
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0155123, upper bound: 1.0258734
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0134487, upper bound: 1.0279677
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3703489, 2.3790803
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7442799, 2.7474465
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3560500, 2.3519416
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2800393, 2.2803760
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8189540, 1.8063908
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2007565, 2.2026334
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1734042, 2.1723418
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7620902, 1.7543640
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9203176, 1.9249375
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0285096, 2.0317857

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276912, upper bound: 1.0132275
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0253989, upper bound: 1.0155267
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3979459, 2.3869786
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7265968, 2.7294927
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3700786, 2.3610811
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2822633, 2.2805686
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8166924, 1.8160565
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2359858, 2.2386570
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1501493, 2.1558661
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7821183, 1.7827733
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9123096, 1.9088585
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0103445, 2.0082107

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0199541, upper bound: 1.0220275
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0144958, upper bound: 1.0274917
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3963084, 2.3886156
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7295322, 2.7265577
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3663297, 2.3648305
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2823930, 2.2804384
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8128624, 1.8198867
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2368708, 2.2377720
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1514621, 2.1545534
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7794290, 1.7854626
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9142838, 1.9068840
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0032949, 2.0152602

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0176526, upper bound: 1.0243266
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0121942, upper bound: 1.0297916
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3596330, 2.3748541
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7281189, 2.7299542
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3441529, 2.3401017
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2795815, 2.2763963
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8263731, 1.8255565
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1957464, 2.1982377
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1556406, 2.1500022
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7479076, 1.7369759
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9220619, 1.9261134
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0478644, 2.0348518

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298497, upper bound: 1.0106959
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275504, upper bound: 1.0130002
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3931475, 2.4014049
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7342348, 2.7322044
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3790874, 2.3845749
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2782068, 2.2779901
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8539801, 1.8574915
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2271037, 2.2232416
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1422863, 2.1318119
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7813692, 1.7807252
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9240255, 1.9281518
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0393896, 2.0338573

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276088, upper bound: 1.0141766
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0221410, upper bound: 1.0196354
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3931475, 2.4014049
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7342358, 2.7322040
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3790874, 2.3845754
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2782068, 2.2779908
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8539791, 1.8574922
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2271042, 2.2232411
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1422873, 2.1318107
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7813687, 1.7807255
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9240260, 1.9281521
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0393891, 2.0338581

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276658, upper bound: 1.0195595
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0253627, upper bound: 1.0218609
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3725591, 2.3619285
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7318993, 2.7261724
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3380704, 2.3461838
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2759027, 2.2800751
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8228540, 1.8290749
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1981950, 2.1957896
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1526847, 2.1529579
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7373543, 1.7475290
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9254370, 1.9227376
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0392485, 2.0434678

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0151627, upper bound: 1.0255288
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0151622, upper bound: 1.0276979
time: 4.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0155123, upper bound: 1.0258734
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0134487, upper bound: 1.0279677
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0276912, upper bound: 1.0132275
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0253989, upper bound: 1.0155267
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0199541, upper bound: 1.0220275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0144958, upper bound: 1.0274917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0176526, upper bound: 1.0243266
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0121942, upper bound: 1.0297916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0298497, upper bound: 1.0106959
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0275504, upper bound: 1.0130002
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0276088, upper bound: 1.0141766
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0221410, upper bound: 1.0196354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0276658, upper bound: 1.0195595
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0253627, upper bound: 1.0218609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0151627, upper bound: 1.0255288
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.96
Output dim: 7, lower bound: -1.0151622, upper bound: 1.0276979

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4367590, 2.4297128
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7310104, 2.7333007
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3459172, 2.3516011
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2659569, 2.2693057
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8117266, 1.8065825
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1886153, 2.1851950
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1919823, 2.1995366
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7777612, 1.7833006
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9466791, 1.9471552
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0595722, 2.0681539

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0134485, upper bound: 1.0256647
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0111984, upper bound: 1.0279636
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3588839, 2.3659782
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7208080, 2.7269106
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3298035, 2.3219447
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2810683, 2.2815347
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7948022, 1.7784085
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2077603, 2.2105219
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1686459, 2.1688957
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7490473, 1.7386322
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9052844, 1.9118783
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0159011, 2.0121279

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276289, upper bound: 1.0076992
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0221717, upper bound: 1.0131686
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4006925, 2.3893790
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7279873, 2.7307072
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3735428, 2.3650823
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2827358, 2.2811069
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8173923, 1.8168736
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2389169, 2.2411866
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1484928, 2.1544168
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7830100, 1.7838132
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9129586, 1.9095960
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0099592, 2.0073519

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0144957, upper bound: 1.0254541
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0124152, upper bound: 1.0274922
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3990560, 2.3910160
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7309227, 2.7277722
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3697939, 2.3688312
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2828655, 2.2809772
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8135624, 1.8207037
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2398019, 2.2403018
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1498060, 2.1531043
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7803216, 1.7865026
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9149332, 1.9076214
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0029092, 2.0144014

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0121942, upper bound: 1.0277045
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101651, upper bound: 1.0297939
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0298497, upper bound: 1.0106958
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276810, upper bound: 1.0106961
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0274882, upper bound: 1.0074719
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0220237, upper bound: 1.0129381
time: 4.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3955479, 2.4041524
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7354488, 2.7335935
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3830886, 2.3880386
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2787457, 2.2784626
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8547969, 1.8581913
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2296343, 2.2261734
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1408377, 2.1301563
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7824092, 1.7816179
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9247642, 1.9288020
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0385308, 2.0334716

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0209262, upper bound: 1.0141646
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276005, upper bound: 1.0074999
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3816795, 2.3883009
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7107573, 2.7116618
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3528323, 2.3545709
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2792363, 2.2791502
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8298316, 1.8295147
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2341089, 2.2311308
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1375308, 2.1283672
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7683306, 1.7649987
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9089899, 1.9150903
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0267954, 2.0142150

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0209835, upper bound: 1.0195514
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276573, upper bound: 1.0128946
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3725591, 2.3619285
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7318964, 2.7261691
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3380675, 2.3461814
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2758975, 2.2800705
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8228550, 1.8290763
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1981950, 2.1957896
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1526771, 2.1529484
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7373540, 1.7475288
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9254375, 1.9227374
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0392504, 2.0434706

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6135
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0151579, upper bound: 1.0258584
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0133326, upper bound: 1.0276938
time: 4.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0134485, upper bound: 1.0256647
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0111984, upper bound: 1.0279636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0276289, upper bound: 1.0076992
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0221717, upper bound: 1.0131686
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0144957, upper bound: 1.0254541
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0124152, upper bound: 1.0274922
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0121942, upper bound: 1.0277045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0101651, upper bound: 1.0297939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0298497, upper bound: 1.0106958
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0276810, upper bound: 1.0106961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0274882, upper bound: 1.0074719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0220237, upper bound: 1.0129381
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0209262, upper bound: 1.0141646
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0276005, upper bound: 1.0074999
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0209835, upper bound: 1.0195514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0276573, upper bound: 1.0128946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0151579, upper bound: 1.0258584
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.87
Output dim: 7, lower bound: -1.0133326, upper bound: 1.0276938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4236574, 2.4182491
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7104740, 2.7098293
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3159204, 2.3253531
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2671170, 2.2703352
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7837439, 1.7824295
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.1965036, 2.1921983
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1885381, 2.1947794
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7620296, 1.7702587
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9336200, 1.9321222
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0399146, 2.0555456

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 6192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0110465, upper bound: 1.0257718
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0109992, upper bound: 1.0279662
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.3612843, 2.3687258
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7220230, 2.7282996
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3338022, 2.3254085
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2816076, 2.2820077
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.7956190, 1.7791080
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2102909, 2.2134540
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1671977, 2.1672406
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7500877, 1.7395246
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9060221, 1.9125276
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0150433, 2.0117431

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 451
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 451

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0276283, upper bound: 1.0054946
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0254357, upper bound: 1.0076988
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.8640556, -11.6210804, -15.8640556, -11.6210804, -2.4006920, 2.3893790
1: -7.1961079, -4.3803363, -7.1961079, -4.3803363, -2.7279854, 2.7307048
2: -8.7408409, -6.1397600, -8.7408409, -6.1397600, -2.3735390, 2.3650789
3: -5.0245595, -2.4202366, -5.0245595, -2.4202366, -2.2827311, 2.2811027
4: -7.9703813, -5.2681599, -7.9703813, -5.2681599, -1.8173866, 1.8168688
5: -6.3388715, -3.7086442, -6.3388715, -3.7086442, -2.2389145, 2.2411838
6: -14.4134359, -10.9648418, -14.4134359, -10.9648418, -2.1484852, 2.1544075
7: 2.2540903, 4.8381100, 2.2540903, 4.8381100, -1.7830079, 1.7838113
8: -1.3332825, 0.9782434, -1.3332825, 0.9782434, -1.9129601, 1.9095974
9: -8.8183250, -5.7160292, -8.8183250, -5.7160292, -2.0099540, 2.0073473

Time for backsubstitution: 14.00 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8196511268615723
rel_dist={7: [-1.029880698625119, 1.029879313881131]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2414.16 seconds
