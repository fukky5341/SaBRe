## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.20377202038
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.8367071, 3.8367071)
1: (-13.2111492, -8.7825651, -13.2111492, -8.7825651, -4.4285841, 4.4285841)
2: (-8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100)
3: (-9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.6361666, 4.6361666)
4: (-11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687)
5: (-0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4574890, 3.4574890)
6: (4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096)
7: (-18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.7471981, 3.7471981)
8: (0.0874861, 4.0993404, 0.0874861, 4.0993404, -4.0118542, 4.0118542)
9: (-8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.1832142, 3.1832142)

## BASE Result
execution time: IAR + LP analysis = 14.91 + 32.30 = 47.21 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.79 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.050609588623047
rel_dist={6: [-1.840513682283781, 1.840511023435588]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.050609588623047
rel_dist={6: [-1.4762376626074394, 1.476239686090869]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.956814765930176
rel_dist={6: [-1.2041204236983045, 1.204119556210955]}

## Binary Search Result
Binary search time: 152.79 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 3400.00 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8405041, upper bound: 1.8333084
time: 6.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8405041, upper bound: 1.8405034
time: 6.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.35 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.35
Output dim: 6, lower bound: -1.8405041, upper bound: 1.8333084
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.35
Output dim: 6, lower bound: -1.8405041, upper bound: 1.8405034

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.5213375, -5.7259974, -9.5402212, -5.7111201, -3.4965162, 3.6284752
1: -13.1510839, -8.8311186, -13.1963568, -8.7851028, -3.9514661, 3.9860511
2: -8.1023321, -4.3769217, -8.1244526, -4.3423529, -3.7599792, 3.7475309
3: -9.7492733, -5.2276101, -9.7977085, -5.1825533, -4.4234705, 4.4255896
4: -10.9811316, -7.1514211, -11.0620813, -7.1014509, -3.8768339, 3.9106603
5: -0.2204313, 3.1744542, -0.2586818, 3.1893830, -3.3757601, 3.4312525
6: 4.5116096, 7.4887538, 4.4701891, 7.5074892, -2.8870873, 3.0185647
7: -18.0245476, -14.3196735, -18.0373535, -14.2973080, -3.4869919, 3.4931903
8: 0.1359761, 4.0514793, 0.0994359, 4.0956306, -3.9383011, 3.8703928
9: -8.8190193, -5.7852249, -8.8979340, -5.7404695, -2.9355297, 2.9010012

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351025, upper bound: 1.8332919
time: 4.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404869, upper bound: 1.8332915
time: 6.11 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.5452709, -5.7085772, -9.5452747, -5.7085748, -3.6551661, 3.6459818
1: -13.2111359, -8.7825670, -13.2111483, -8.7825680, -4.0216246, 4.0364828
2: -8.1306238, -4.3364310, -8.1306276, -4.3364244, -3.7941995, 3.7941966
3: -9.8012638, -5.1651335, -9.8012695, -5.1651130, -4.4960117, 4.4608665
4: -11.0695486, -7.0786190, -11.0695572, -7.0785999, -3.9909487, 3.9909382
5: -0.2625251, 3.1949496, -0.2625289, 3.1949582, -3.4270639, 3.4161811
6: 4.4642420, 7.5148306, 4.4642358, 7.5148377, -3.0505958, 3.0505948
7: -18.0411339, -14.2939463, -18.0411415, -14.2939453, -3.5046310, 3.5086427
8: 0.0875016, 4.0993347, 0.0874898, 4.0993395, -3.9703312, 3.9922905
9: -8.9012642, -5.7180891, -8.9012680, -5.7180672, -2.9954424, 2.9483800

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351025, upper bound: 1.8404867
time: 5.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404869, upper bound: 1.8404866
time: 7.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.99 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.99
Output dim: 6, lower bound: -1.8351025, upper bound: 1.8332919
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.99
Output dim: 6, lower bound: -1.8404869, upper bound: 1.8332915
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.99
Output dim: 6, lower bound: -1.8351025, upper bound: 1.8404867
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.99
Output dim: 6, lower bound: -1.8404869, upper bound: 1.8404866

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.5189943, -5.7264242, -9.5305119, -5.7130961, -3.4890990, 3.4982495
1: -13.1449118, -8.8327188, -13.1698341, -8.7922564, -3.9576492, 3.9536490
2: -8.0934944, -4.3799791, -8.0865803, -4.3561406, -3.7373538, 3.7066011
3: -9.7484436, -5.2294617, -9.7941074, -5.1909022, -4.4037075, 4.4100933
4: -10.9795189, -7.1537895, -11.0538664, -7.1115918, -3.8652143, 3.9000769
5: -0.2187567, 3.1723137, -0.2512910, 3.1800675, -3.3749533, 3.4217868
6: 4.5132341, 7.4862537, 4.4774761, 7.4969444, -2.8751607, 3.0087776
7: -18.0207367, -14.3215446, -18.0210934, -14.3059187, -3.4856901, 3.4755111
8: 0.1373990, 4.0472083, 0.1056869, 4.0772791, -3.8550987, 3.8604717
9: -8.8173714, -5.7904625, -8.8903379, -5.7629409, -2.9123635, 2.9420276

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350997, upper bound: 1.8278970
time: 4.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350997, upper bound: 1.8332919
time: 4.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.5213242, -5.7260008, -9.5775900, -5.6986771, -3.5074615, 3.6715989
1: -13.1510458, -8.8311253, -13.2323370, -8.7172050, -3.9975500, 4.0332851
2: -8.1022720, -4.3769369, -8.1390791, -4.2240829, -3.8781891, 3.7621422
3: -9.7492695, -5.2276182, -9.8170738, -5.1056757, -4.5293608, 4.4414768
4: -10.9811249, -7.1514349, -11.1097136, -7.0924349, -3.8868408, 3.9582787
5: -0.2204235, 3.1744423, -0.3070841, 3.1983056, -3.3849192, 3.4815264
6: 4.5116200, 7.4887419, 4.4206796, 7.5182543, -2.8993702, 3.0680623
7: -18.0245228, -14.3196840, -18.0515385, -14.2553425, -3.5349894, 3.5062728
8: 0.1359836, 4.0514669, 0.0391641, 4.1158037, -3.9581604, 3.9169412
9: -8.8190117, -5.7852564, -8.9645824, -5.7303014, -2.9396830, 2.9565928

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8278967
time: 5.10 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8332916
time: 5.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.5424948, -5.7090344, -9.5346823, -5.7105408, -3.6470423, 3.5114422
1: -13.2049398, -8.7842846, -13.1846352, -8.7899237, -4.0248480, 4.0032349
2: -8.1217842, -4.3396692, -8.0927324, -4.3504038, -3.7713804, 3.7530632
3: -9.8003979, -5.1670647, -9.7975979, -5.1735396, -4.4761257, 4.4443021
4: -11.0672541, -7.0809927, -11.0601006, -7.0887489, -3.9785051, 3.9791079
5: -0.2607911, 3.1927085, -0.2551103, 3.1854446, -3.4257812, 3.4070711
6: 4.4659438, 7.5123467, 4.4715500, 7.5042720, -3.0383282, 3.0407968
7: -18.0373325, -14.2959881, -18.0248871, -14.3027725, -3.5066223, 3.4903488
8: 0.0889474, 4.0949593, 0.0937333, 4.0807037, -3.8799505, 3.9816298
9: -8.8994036, -5.7233315, -8.8935957, -5.7405186, -2.9712124, 2.9558640

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350996, upper bound: 1.8350993
time: 4.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350996, upper bound: 1.8404868
time: 5.03 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.5452566, -5.7085814, -9.5826006, -5.6961145, -3.6665654, 3.6853609
1: -13.2110968, -8.7825766, -13.2471218, -8.7146473, -4.0522833, 4.0710812
2: -8.1305599, -4.3364444, -8.1452112, -4.2182093, -3.9123507, 3.8087668
3: -9.8012609, -5.1651421, -9.8206425, -5.0881319, -4.5835986, 4.4767723
4: -11.0695372, -7.0786362, -11.1171808, -7.0695987, -3.9999385, 4.0385447
5: -0.2625172, 3.1949418, -0.3109803, 3.2038727, -3.4362583, 3.4793835
6: 4.4642496, 7.5148172, 4.4145689, 7.5255566, -3.0613070, 3.1002483
7: -18.0411091, -14.2939568, -18.0553303, -14.2519760, -3.5526218, 3.5219831
8: 0.0875076, 4.0993223, 0.0272713, 4.1195230, -3.9901648, 4.0227499
9: -8.9012547, -5.7181206, -8.9681702, -5.7078571, -2.9996958, 3.0015087

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8350988
time: 5.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8404865
time: 5.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.26 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8350997, upper bound: 1.8278970
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8350997, upper bound: 1.8332919
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8278967
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8332916
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8350996, upper bound: 1.8350993
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8350996, upper bound: 1.8404868
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8350988
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 6, lower bound: -1.8404872, upper bound: 1.8404865

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.5121298, -5.7278528, -9.5305119, -5.7130961, -3.4797721, 3.4931278
1: -13.1246319, -8.8380508, -13.1698341, -8.7922564, -3.9354630, 3.9484363
2: -8.0644293, -4.3901663, -8.0865803, -4.3561406, -3.7082887, 3.6964140
3: -9.7457447, -5.2356782, -9.7941074, -5.1909022, -4.3953009, 4.3972058
4: -10.9741535, -7.1615572, -11.0538664, -7.1115918, -3.8601160, 3.8923092
5: -0.2134156, 3.1652951, -0.2512910, 3.1800675, -3.3694220, 3.4147081
6: 4.5186205, 7.4781103, 4.4774761, 7.4969444, -2.8691578, 3.0006342
7: -18.0082588, -14.3278236, -18.0210934, -14.3059187, -3.4727154, 3.4707470
8: 0.1421187, 4.0331779, 0.1056869, 4.0772791, -3.8514347, 3.8440800
9: -8.8118515, -5.8076653, -8.8903379, -5.7629409, -2.9082284, 2.9246247

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8278970
time: 4.74 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8278967
time: 4.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.5587301, -5.7147589, -9.5305119, -5.7130961, -3.5267243, 3.5087519
1: -13.1854315, -8.7632751, -13.1698341, -8.7922564, -3.9979162, 3.9982715
2: -8.1173630, -4.2595696, -8.0865803, -4.3561406, -3.7612224, 3.8270106
3: -9.7685747, -5.1511712, -9.7941074, -5.1909022, -4.4223061, 4.4950027
4: -11.0241747, -7.1422501, -11.0538664, -7.1115918, -3.9125829, 3.9116163
5: -0.2689233, 3.1824064, -0.2512910, 3.1800675, -3.4402189, 3.4325323
6: 4.4649634, 7.4998865, 4.4774761, 7.4969444, -2.9250221, 3.0224104
7: -18.0387745, -14.2782593, -18.0210934, -14.3059187, -3.5027084, 3.5218501
8: 0.0755713, 4.0717306, 0.1056869, 4.0772791, -3.9075980, 3.8829327
9: -8.8852348, -5.7757912, -8.8903379, -5.7629409, -2.9579992, 2.9554648

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8332919
time: 5.34 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8332915
time: 4.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5121298, -5.7278528, -9.5775900, -5.6986771, -3.4951000, 3.6600647
1: -13.1246319, -8.8380508, -13.2323370, -8.7172050, -3.9691401, 4.0126390
2: -8.0644293, -4.3901663, -8.1390791, -4.2240829, -3.8403463, 3.7489128
3: -9.7457447, -5.2356782, -9.8170738, -5.1056757, -4.4937334, 4.4246030
4: -10.9741535, -7.1615572, -11.1097136, -7.0924349, -3.8816414, 3.9481564
5: -0.2134156, 3.1652951, -0.3070841, 3.1983056, -3.3789005, 3.4723792
6: 4.5186205, 7.4781103, 4.4206796, 7.5182543, -2.8934741, 3.0574307
7: -18.0082588, -14.3278236, -18.0515385, -14.2553425, -3.5155172, 3.5000739
8: 0.1421187, 4.0331779, 0.0391641, 4.1158037, -3.9519739, 3.8955650
9: -8.8118515, -5.8076653, -8.9645824, -5.7303014, -2.9390550, 2.9348273

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278992, upper bound: 1.8278968
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278992, upper bound: 1.8278968
time: 5.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5587301, -5.7147589, -9.5775900, -5.6986771, -3.5500698, 3.6810479
1: -13.1854315, -8.7632751, -13.2323370, -8.7172050, -4.0224557, 4.0586472
2: -8.1173630, -4.2595696, -8.1390791, -4.2240829, -3.8932800, 3.8795094
3: -9.7685747, -5.1511712, -9.8170738, -5.1056757, -4.5432863, 4.5450788
4: -11.0241747, -7.1422501, -11.1097136, -7.0924349, -3.9255047, 3.9674635
5: -0.2689233, 3.1824064, -0.3070841, 3.1983056, -3.4454193, 3.4894905
6: 4.4649634, 7.4998865, 4.4206796, 7.5182543, -2.9379153, 3.0792069
7: -18.0387745, -14.2782593, -18.0515385, -14.2553425, -3.5496292, 3.5536699
8: 0.0755713, 4.0717306, 0.0391641, 4.1158037, -3.9960365, 3.9279904
9: -8.8852348, -5.7757912, -8.9645824, -5.7303014, -2.9805250, 2.9476650

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278992, upper bound: 1.8332918
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278991, upper bound: 1.8332915
time: 4.89 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5346832, -5.7105441, -9.5346823, -5.7105408, -3.5040379, 3.5080709
1: -13.1846237, -8.7899265, -13.1846352, -8.7899237, -4.0011396, 4.0089931
2: -8.0927258, -4.3504105, -8.0927324, -4.3504038, -3.7423220, 3.7423220
3: -9.7975931, -5.1735592, -9.7975979, -5.1735396, -4.4670887, 4.4311275
4: -11.0600967, -7.0887690, -11.0601006, -7.0887489, -3.9713478, 3.9713316
5: -0.2551076, 3.1854396, -0.2551103, 3.1854446, -3.4200077, 3.4067450
6: 4.4715552, 7.5042639, 4.4715500, 7.5042720, -3.0327168, 3.0327139
7: -18.0248871, -14.3027763, -18.0248871, -14.3027725, -3.4937258, 3.4927816
8: 0.0937436, 4.0807009, 0.0937333, 4.0807037, -3.8761473, 3.9047966
9: -8.8935919, -5.7405415, -8.8935957, -5.7405186, -2.9888945, 2.9390981

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279019, upper bound: 1.8350989
time: 4.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279000, upper bound: 1.8350991
time: 4.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5825949, -5.6961136, -9.5346823, -5.7105408, -3.6868315, 3.5234766
1: -13.2471104, -8.7146511, -13.1846352, -8.7899237, -4.0653591, 4.0216165
2: -8.1452036, -4.2182150, -8.0927324, -4.3504038, -3.7947998, 3.8745174
3: -9.8206396, -5.0881529, -9.7975979, -5.1735396, -4.4946766, 4.5294981
4: -11.1171722, -7.0696182, -11.0601006, -7.0887489, -4.0284233, 3.9904823
5: -0.3109763, 3.2038672, -0.2551103, 3.1854446, -3.4860816, 3.4188838
6: 4.4145756, 7.5255504, 4.4715500, 7.5042720, -3.0896964, 3.0540004
7: -18.0553246, -14.2519789, -18.0248871, -14.3027725, -3.5238037, 3.5371938
8: 0.0272832, 4.1195188, 0.0937333, 4.0807037, -3.9291096, 4.0050917
9: -8.9681644, -5.7078791, -8.8935957, -5.7405186, -3.0015407, 2.9694772

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279001, upper bound: 1.8404862
time: 5.08 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8279000, upper bound: 1.8404872
time: 4.99 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5346832, -5.7105441, -9.5826006, -5.6961145, -3.5194421, 3.6775360
1: -13.1846237, -8.7899265, -13.2471218, -8.7146473, -4.0209560, 4.0688486
2: -8.0927258, -4.3504105, -8.1452112, -4.2182093, -3.8745165, 3.7948008
3: -9.7975931, -5.1735592, -9.8206425, -5.0881319, -4.5644989, 4.4595323
4: -11.0600967, -7.0887690, -11.1171808, -7.0695987, -3.9904981, 4.0284119
5: -0.2551076, 3.1854396, -0.3109803, 3.2038727, -3.4299264, 3.4774418
6: 4.4715552, 7.5042639, 4.4145689, 7.5255566, -3.0540013, 3.0896950
7: -18.0248871, -14.3027763, -18.0553303, -14.2519760, -3.5332518, 3.5227041
8: 0.0937436, 4.0807009, 0.0272713, 4.1195230, -3.9838524, 3.9362345
9: -8.8935919, -5.7405415, -8.9681702, -5.7078571, -3.0140300, 2.9788065

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8350992
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8350994
time: 5.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.5825949, -5.6961136, -9.5826006, -5.6961145, -3.7044702, 3.6953230
1: -13.2471104, -8.7146511, -13.2471218, -8.7146473, -4.0870781, 4.0876656
2: -8.1452036, -4.2182150, -8.1452112, -4.2182093, -3.9269943, 3.9269962
3: -9.8206396, -5.0881529, -9.8206425, -5.0881319, -4.5974889, 4.5805712
4: -11.1171722, -7.0696182, -11.1171808, -7.0695987, -4.0475736, 4.0475626
5: -0.3109763, 3.2038672, -0.3109803, 3.2038727, -3.4953561, 3.4849782
6: 4.4145756, 7.5255504, 4.4145689, 7.5255566, -3.1109810, 3.1109815
7: -18.0553246, -14.2519789, -18.0553303, -14.2519760, -3.5674458, 3.5712380
8: 0.0272832, 4.1195188, 0.0272713, 4.1195230, -4.0278397, 4.0384293
9: -8.9681644, -5.7078791, -8.9681702, -5.7078571, -3.0285077, 2.9949517

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8404863
time: 5.11 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8404867
time: 4.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.10 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8278970
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8278967
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8332919
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279021, upper bound: 1.8332915
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278992, upper bound: 1.8278968
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278992, upper bound: 1.8278968
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278992, upper bound: 1.8332918
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278991, upper bound: 1.8332915
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279019, upper bound: 1.8350989
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279000, upper bound: 1.8350991
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279001, upper bound: 1.8404862
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8279000, upper bound: 1.8404872
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8350992
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8350994
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8404863
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.10
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8404867

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5121298, -5.7278528, -9.5121298, -5.7278528, -3.4746256, 3.4746256
1: -13.1246319, -8.8380508, -13.1246319, -8.8380508, -3.8873749, 3.8873744
2: -8.0644293, -4.3901663, -8.0644293, -4.3901663, -3.6742630, 3.6742630
3: -9.7457447, -5.2356782, -9.7457447, -5.2356782, -4.3471813, 4.3471813
4: -10.9741535, -7.1615572, -10.9741535, -7.1615572, -3.8097954, 3.8097954
5: -0.2134156, 3.1652951, -0.2134156, 3.1652951, -3.3744669, 3.3744669
6: 4.5186205, 7.4781103, 4.5186205, 7.4781103, -2.8516665, 2.8516660
7: -18.0082588, -14.3278236, -18.0082588, -14.3278236, -3.4538755, 3.4538751
8: 0.1421187, 4.0331779, 0.1421187, 4.0331779, -3.8062963, 3.8062963
9: -8.8118515, -5.8076653, -8.8118515, -5.8076653, -2.8398933, 2.8398933

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235767, upper bound: 1.8278871
time: 5.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278886, upper bound: 1.8278867
time: 9.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5121298, -5.7278528, -9.5346336, -5.7105494, -3.4801841, 3.4970603
1: -13.1246319, -8.8380508, -13.1845303, -8.7899771, -3.9347596, 3.9633803
2: -8.0644293, -4.3901663, -8.0923729, -4.3504233, -3.7140059, 3.7022066
3: -9.7457447, -5.2356782, -9.7975912, -5.1737547, -4.4128342, 4.4009638
4: -10.9741535, -7.1615572, -11.0600357, -7.0887718, -3.8836164, 3.8984785
5: -0.2134156, 3.1652951, -0.2550690, 3.1849971, -3.3743839, 3.4191561
6: 4.5186205, 7.4781103, 4.4731483, 7.5042639, -2.8787827, 3.0049620
7: -18.0082588, -14.3278236, -18.0248146, -14.3027821, -3.4750118, 3.4712787
8: 0.1421187, 4.0331779, 0.0937839, 4.0806913, -3.8544903, 3.8563313
9: -8.8118515, -5.8076653, -8.8935747, -5.7405419, -2.9252100, 2.8921442

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235767, upper bound: 1.8278866
time: 4.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278886, upper bound: 1.8278864
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.5587301, -5.7147589, -9.5121298, -5.7278528, -3.5240641, 3.4902496
1: -13.1854315, -8.7632751, -13.1246319, -8.8380508, -3.9498272, 3.9406478
2: -8.1173630, -4.2595696, -8.0644293, -4.3901663, -3.7271967, 3.8048596
3: -9.7685747, -5.1511712, -9.7457447, -5.2356782, -4.3741875, 4.4451494
4: -11.0241747, -7.1422501, -10.9741535, -7.1615572, -3.8624735, 3.8314314
5: -0.2689233, 3.1824064, -0.2134156, 3.1652951, -3.4342184, 3.3922915
6: 4.4649634, 7.4998865, 4.5186205, 7.4781103, -2.9075308, 2.8770270
7: -18.0387745, -14.2782593, -18.0082588, -14.3278236, -3.4830513, 3.5049787
8: 0.0755713, 4.0717306, 0.1421187, 4.0331779, -3.8624430, 3.8451505
9: -8.8852348, -5.7757912, -8.8118515, -5.8076653, -2.8952298, 2.8707333

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235750, upper bound: 1.8332767
time: 5.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278869, upper bound: 1.8332765
time: 4.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.5587301, -5.7147589, -9.5346336, -5.7105494, -3.5271354, 3.5126843
1: -13.1854315, -8.7632751, -13.1845303, -8.7899771, -3.9930844, 3.9994988
2: -8.1173630, -4.2595696, -8.0923729, -4.3504233, -3.7669396, 3.8328032
3: -9.7685747, -5.1511712, -9.7975912, -5.1737547, -4.4398394, 4.4985557
4: -11.0241747, -7.1422501, -11.0600357, -7.0887718, -3.9216914, 3.9177856
5: -0.2689233, 3.1824064, -0.2550690, 3.1849971, -3.4450350, 3.4369802
6: 4.4649634, 7.4998865, 4.4731483, 7.5042639, -2.9346471, 3.0267382
7: -18.0387745, -14.2782593, -18.0248146, -14.3027821, -3.5050049, 3.5223813
8: 0.0755713, 4.0717306, 0.0937839, 4.0806913, -3.8859873, 3.8951836
9: -8.8852348, -5.7757912, -8.8935747, -5.7405419, -2.9599748, 2.9172633

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235750, upper bound: 1.8332765
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278869, upper bound: 1.8332763
time: 5.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5121298, -5.7278528, -9.5587301, -5.7136745, -3.4778728, 3.5240636
1: -13.1246319, -8.8380508, -13.1872225, -8.7632751, -3.9406476, 3.9667420
2: -8.0644293, -4.3901663, -8.1173630, -4.2584009, -3.8060284, 3.7271967
3: -9.7457447, -5.2356782, -9.7686481, -5.1511712, -4.4451485, 4.3740983
4: -10.9741535, -7.1615572, -11.0284958, -7.1422501, -3.8314304, 3.8669386
5: -0.2134156, 3.1652951, -0.2689233, 3.1834569, -3.3743572, 3.4342184
6: 4.5186205, 7.4781103, 4.4625912, 7.4998865, -2.8770270, 3.0155191
7: -18.0082588, -14.3278236, -18.0387745, -14.2776575, -3.4997768, 3.4830513
8: 0.1421187, 4.0331779, 0.0755663, 4.0717306, -3.8451509, 3.8626661
9: -8.8118515, -5.8076653, -8.8852348, -5.7752943, -2.8946495, 2.8952293

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8289665, upper bound: 1.8278852
time: 5.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332784, upper bound: 1.8278846
time: 5.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5121298, -5.7278528, -9.5825481, -5.6961212, -3.4955878, 3.6541071
1: -13.1246319, -8.8380508, -13.2470150, -8.7147026, -3.9426808, 4.0274673
2: -8.0644293, -4.3901663, -8.1448536, -4.2182274, -3.8462019, 3.7546873
3: -9.7457447, -5.2356782, -9.8206406, -5.0883479, -4.5113764, 4.4284601
4: -10.9741535, -7.1615572, -11.1171131, -7.0696211, -3.9045324, 3.9555559
5: -0.2134156, 3.1652951, -0.3109388, 3.2034259, -3.3845782, 3.4762340
6: 4.5186205, 7.4781103, 4.4161668, 7.5255499, -2.9030252, 3.0619435
7: -18.0082588, -14.3278236, -18.0552540, -14.2519875, -3.5196877, 3.5005169
8: 0.1421187, 4.0331779, 0.0273249, 4.1195078, -3.9544315, 3.8987994
9: -8.8118515, -5.8076653, -8.9681501, -5.7078795, -2.9502158, 2.8828962

Time for backsubstitution: 15.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8289665, upper bound: 1.8278851
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332784, upper bound: 1.8278848
time: 5.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5587301, -5.7147589, -9.5587301, -5.7136745, -3.5330052, 3.5527048
1: -13.1854315, -8.7632751, -13.1872225, -8.7632751, -3.9938965, 4.0117431
2: -8.1173630, -4.2595696, -8.1173630, -4.2584009, -3.8589621, 3.8577933
3: -9.7685747, -5.1511712, -9.7686481, -5.1511712, -4.4947033, 4.4941468
4: -11.0241747, -7.1422501, -11.0284958, -7.1422501, -3.8752937, 3.8862457
5: -0.2689233, 3.1824064, -0.2689233, 3.1834569, -3.4406366, 3.4513297
6: 4.4649634, 7.4998865, 4.4625912, 7.4998865, -2.9214349, 3.0372953
7: -18.0387745, -14.2782593, -18.0387745, -14.2776575, -3.5351543, 3.5366478
8: 0.0755713, 4.0717306, 0.0755663, 4.0717306, -3.8901682, 3.8903580
9: -8.8852348, -5.7757912, -8.8852348, -5.7752943, -2.9359512, 2.9140425

Time for backsubstitution: 15.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235720, upper bound: 1.8332773
time: 5.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278839, upper bound: 1.8332769
time: 5.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.5587301, -5.7147589, -9.5825481, -5.6961212, -3.5505705, 3.6750913
1: -13.1854315, -8.7632751, -13.2470150, -8.7147026, -4.0041137, 4.0649090
2: -8.1173630, -4.2595696, -8.1448536, -4.2182274, -3.8991356, 3.8852839
3: -9.7685747, -5.1511712, -9.8206406, -5.0883479, -4.5544276, 4.5314665
4: -11.0241747, -7.1422501, -11.1171131, -7.0696211, -3.9420753, 3.9748631
5: -0.2689233, 3.1824064, -0.3109388, 3.2034259, -3.4510984, 3.4933453
6: 4.4649634, 7.4998865, 4.4161668, 7.5255499, -2.9474664, 3.0837197
7: -18.0387745, -14.2782593, -18.0552540, -14.2519875, -3.5537987, 3.5541129
8: 0.0755713, 4.0717306, 0.0273249, 4.1195078, -3.9880114, 3.9357877
9: -8.8852348, -5.7757912, -8.9681501, -5.7078795, -2.9852705, 2.9086452

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235720, upper bound: 1.8332769
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278840, upper bound: 1.8332768
time: 5.18 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5346336, -5.7105494, -9.5121298, -5.7278528, -3.4970598, 3.4801836
1: -13.1845303, -8.7899771, -13.1246319, -8.8380508, -3.9633808, 3.9347601
2: -8.0923729, -4.3504233, -8.0644293, -4.3901663, -3.7022066, 3.7140059
3: -9.7975912, -5.1737547, -9.7457447, -5.2356782, -4.4009638, 4.4128342
4: -11.0600357, -7.0887718, -10.9741535, -7.1615572, -3.8984785, 3.8836164
5: -0.2550690, 3.1849971, -0.2134156, 3.1652951, -3.4191561, 3.3743834
6: 4.4731483, 7.5042639, 4.5186205, 7.4781103, -3.0049620, 2.8787827
7: -18.0248146, -14.3027821, -18.0082588, -14.3278236, -3.4712782, 3.4750128
8: 0.0937839, 4.0806913, 0.1421187, 4.0331779, -3.8563318, 3.8544903
9: -8.8935747, -5.7405419, -8.8118515, -5.8076653, -2.8921447, 2.9252100

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235746, upper bound: 1.8350890
time: 5.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278866, upper bound: 1.8350886
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5346832, -5.7105441, -9.5346832, -5.7105441, -3.5080652, 3.5080652
1: -13.1846237, -8.7899265, -13.1846237, -8.7899265, -4.0011339, 4.0011339
2: -8.0927258, -4.3504105, -8.0927258, -4.3504105, -3.7423153, 3.7423153
3: -9.7975931, -5.1735592, -9.7975931, -5.1735592, -4.4311237, 4.4311237
4: -11.0600967, -7.0887690, -11.0600967, -7.0887690, -3.9713278, 3.9713278
5: -0.2551076, 3.1854396, -0.2551076, 3.1854396, -3.4067411, 3.4067411
6: 4.4715552, 7.5042639, 4.4715552, 7.5042639, -3.0327086, 3.0327086
7: -18.0248871, -14.3027763, -18.0248871, -14.3027763, -3.4937210, 3.4937201
8: 0.0937436, 4.0807009, 0.0937436, 4.0807009, -3.8761425, 3.8761425
9: -8.8935919, -5.7405415, -8.8935919, -5.7405415, -2.9390950, 2.9390953

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235746, upper bound: 1.8350890
time: 5.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278865, upper bound: 1.8350888
time: 5.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.5825481, -5.6961212, -9.5121298, -5.7278528, -3.6541071, 3.4955883
1: -13.2470150, -8.7147026, -13.1246319, -8.8380508, -4.0274677, 3.9426811
2: -8.1448536, -4.2182274, -8.0644293, -4.3901663, -3.7546873, 3.8462019
3: -9.8206406, -5.0883479, -9.7457447, -5.2356782, -4.4284611, 4.5113769
4: -11.1171131, -7.0696211, -10.9741535, -7.1615572, -3.9555559, 3.9045324
5: -0.3109388, 3.2034259, -0.2134156, 3.1652951, -3.4762340, 3.3845782
6: 4.4161668, 7.5255499, 4.5186205, 7.4781103, -3.0619435, 2.9030254
7: -18.0552540, -14.2519875, -18.0082588, -14.3278236, -3.5005159, 3.5196872
8: 0.0273249, 4.1195078, 0.1421187, 4.0331779, -3.8987989, 3.9544315
9: -8.9681501, -5.7078795, -8.8118515, -5.8076653, -2.8828959, 2.9502158

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235730, upper bound: 1.8404715
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278849, upper bound: 1.8404711
time: 4.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.5825949, -5.6961136, -9.5346832, -5.7105441, -3.6868248, 3.5234714
1: -13.2471104, -8.7146511, -13.1846237, -8.7899265, -4.0653534, 4.0209527
2: -8.1452036, -4.2182150, -8.0927258, -4.3504105, -3.7947931, 3.8745108
3: -9.8206396, -5.0881529, -9.7975931, -5.1735592, -4.4595280, 4.5294943
4: -11.1171722, -7.0696182, -11.0600967, -7.0887690, -4.0284033, 3.9904785
5: -0.3109763, 3.2038672, -0.2551076, 3.1854396, -3.4774389, 3.4188781
6: 4.4145756, 7.5255504, 4.4715552, 7.5042639, -3.0896883, 3.0539951
7: -18.0553246, -14.2519789, -18.0248871, -14.3027763, -3.5237989, 3.5332494
8: 0.0272832, 4.1195188, 0.0937436, 4.0807009, -3.9254541, 3.9838490
9: -8.9681644, -5.7078791, -8.8935919, -5.7405415, -2.9733930, 2.9694743

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235729, upper bound: 1.8404718
time: 5.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278849, upper bound: 1.8404714
time: 5.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5346336, -5.7105494, -9.5587301, -5.7136745, -3.4999876, 3.5271358
1: -13.1845303, -8.7899771, -13.1872225, -8.7632751, -3.9994984, 4.0073376
2: -8.0923729, -4.3504233, -8.1173630, -4.2584009, -3.8339720, 3.7669396
3: -9.7975912, -5.1737547, -9.7686481, -5.1511712, -4.4985552, 4.4397368
4: -11.0600357, -7.0887718, -11.0284958, -7.1422501, -3.9177856, 3.9397240
5: -0.2550690, 3.1849971, -0.2689233, 3.1834569, -3.4195271, 3.4450355
6: 4.4731483, 7.5042639, 4.4625912, 7.4998865, -3.0267382, 3.0416727
7: -18.0248146, -14.3027821, -18.0387745, -14.2776575, -3.5173931, 3.5050058
8: 0.0937839, 4.0806913, 0.0755663, 4.0717306, -3.8951836, 3.8862104
9: -8.8935747, -5.7405419, -8.8852348, -5.7752943, -2.9461684, 2.9599745

Time for backsubstitution: 15.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8289646, upper bound: 1.8350871
time: 6.13 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332764, upper bound: 1.8350865
time: 5.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5346832, -5.7105441, -9.5825949, -5.6961136, -3.5234709, 3.6868253
1: -13.1846237, -8.7899265, -13.2471104, -8.7146511, -4.0209532, 4.0653534
2: -8.0927258, -4.3504105, -8.1452036, -4.2182150, -3.8745108, 3.7947931
3: -9.7975931, -5.1735592, -9.8206396, -5.0881529, -4.5294943, 4.4595284
4: -11.0600967, -7.0887690, -11.1171722, -7.0696182, -3.9904785, 4.0284033
5: -0.2551076, 3.1854396, -0.3109763, 3.2038672, -3.4188776, 3.4774389
6: 4.4715552, 7.5042639, 4.4145756, 7.5255504, -3.0539951, 3.0896883
7: -18.0248871, -14.3027763, -18.0553246, -14.2519789, -3.5332499, 3.5237989
8: 0.0937436, 4.0807009, 0.0272832, 4.1195188, -3.9838495, 3.9254532
9: -8.8935919, -5.7405415, -8.9681644, -5.7078791, -2.9694743, 2.9733930

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8289646, upper bound: 1.8350876
time: 5.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332764, upper bound: 1.8350872
time: 5.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5825481, -5.6961212, -9.5587301, -5.7136745, -3.6708732, 3.5505705
1: -13.2470150, -8.7147026, -13.1872225, -8.7632751, -4.0649090, 4.0258312
2: -8.1448536, -4.2182274, -8.1173630, -4.2584009, -3.8864527, 3.8991356
3: -9.8206406, -5.0883479, -9.7686481, -5.1511712, -4.5314665, 4.5541220
4: -11.1171131, -7.0696211, -11.0284958, -7.1422501, -3.9748631, 3.9588747
5: -0.3109388, 3.2034259, -0.2689233, 3.1834569, -3.4848576, 3.4510984
6: 4.4161668, 7.5255499, 4.4625912, 7.4998865, -3.0837197, 3.0629587
7: -18.0552540, -14.2519875, -18.0387745, -14.2776575, -3.5527029, 3.5537992
8: 0.0273249, 4.1195078, 0.0755663, 4.0717306, -3.9357867, 3.9883032
9: -8.9681501, -5.7078795, -8.8852348, -5.7752943, -2.9611049, 2.9852705

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8235700, upper bound: 1.8404718
time: 5.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278820, upper bound: 1.8404714
time: 5.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.59 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235767, upper bound: 1.8278871
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278886, upper bound: 1.8278867
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235767, upper bound: 1.8278866
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278886, upper bound: 1.8278864
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235750, upper bound: 1.8332767
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278869, upper bound: 1.8332765
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235750, upper bound: 1.8332765
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278869, upper bound: 1.8332763
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8289665, upper bound: 1.8278852
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8332784, upper bound: 1.8278846
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8289665, upper bound: 1.8278851
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8332784, upper bound: 1.8278848
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235720, upper bound: 1.8332773
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278839, upper bound: 1.8332769
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235720, upper bound: 1.8332769
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278840, upper bound: 1.8332768
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235746, upper bound: 1.8350890
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278866, upper bound: 1.8350886
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235746, upper bound: 1.8350890
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278865, upper bound: 1.8350888
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235730, upper bound: 1.8404715
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278849, upper bound: 1.8404711
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235729, upper bound: 1.8404718
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278849, upper bound: 1.8404714
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8289646, upper bound: 1.8350871
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8332764, upper bound: 1.8350865
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8289646, upper bound: 1.8350876
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8332764, upper bound: 1.8350872
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8235700, upper bound: 1.8404718
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.59
Output dim: 6, lower bound: -1.8278820, upper bound: 1.8404714
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.59
Output dim: 6, lower bound: -1.8278971, upper bound: 1.8404867
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.050609588623047
rel_dist={6: [-1.840513682283781, 1.840511023435588]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762098, upper bound: 1.4728713
time: 8.10 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762338, upper bound: 1.4762332
time: 5.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.45 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.45
Output dim: 6, lower bound: -1.4762098, upper bound: 1.4728713
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.45
Output dim: 6, lower bound: -1.4762338, upper bound: 1.4762332

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.5213375, -5.7259974, -9.5367088, -5.7130451, -3.2597561, 3.2714243
1: -13.1510839, -8.8311186, -13.1853390, -8.7869625, -3.6638913, 3.6794562
2: -8.1023321, -4.3769217, -8.1198540, -4.3468027, -3.7555294, 3.7429323
3: -9.7492733, -5.2276101, -9.7950554, -5.1956010, -4.0845613, 4.0964980
4: -10.9811316, -7.1514211, -11.0568104, -7.1185598, -3.5515251, 3.7175107
5: -0.2204313, 3.1744542, -0.2557757, 3.1852856, -3.2134342, 3.2608337
6: 4.5116096, 7.4887538, 4.4746723, 7.5020084, -2.6881099, 3.0140815
7: -18.0245476, -14.3196735, -18.0345230, -14.2998133, -3.1793413, 3.1749377
8: 0.1359761, 4.0514793, 0.1083713, 4.0929203, -3.6972990, 3.6833382
9: -8.8190193, -5.7852249, -8.8955145, -5.7572317, -2.6817627, 2.7086384

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733158, upper bound: 1.4728614
time: 5.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4761998, upper bound: 1.4728612
time: 6.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.5452709, -5.7085772, -9.5452738, -5.7085762, -3.4094648, 3.4001732
1: -13.2111359, -8.7825670, -13.2111406, -8.7825689, -3.7101707, 3.7301230
2: -8.1306238, -4.3364310, -8.1306267, -4.3364277, -3.7941961, 3.7941957
3: -9.8012638, -5.1651335, -9.8012695, -5.1651220, -4.1698580, 4.1288614
4: -11.0695486, -7.0786190, -11.0695524, -7.0786076, -3.8401918, 3.8060999
5: -0.2625251, 3.1949496, -0.2625272, 3.1949546, -3.2596111, 3.2460284
6: 4.4642420, 7.5148306, 4.4642377, 7.5148363, -3.0505943, 3.0505929
7: -18.0411339, -14.2939463, -18.0411358, -14.2939453, -3.1892719, 3.1944480
8: 0.0875016, 4.0993347, 0.0874958, 4.0993381, -3.7954197, 3.8188019
9: -8.9012642, -5.7180891, -8.9012661, -5.7180758, -2.7736931, 2.7179751

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733612, upper bound: 1.4762224
time: 5.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762237, upper bound: 1.4762233
time: 5.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.99 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 6, lower bound: -1.4733158, upper bound: 1.4728614
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 6, lower bound: -1.4761998, upper bound: 1.4728612
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 6, lower bound: -1.4733612, upper bound: 1.4762224
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.99
Output dim: 6, lower bound: -1.4762237, upper bound: 1.4762233

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.5161409, -5.7270184, -9.5273972, -5.7150245, -3.2481785, 3.2530055
1: -13.1364431, -8.8349361, -13.1588154, -8.7940178, -3.6404343, 3.6449218
2: -8.0813675, -4.3842087, -8.0819988, -4.3604851, -3.7208824, 3.6977901
3: -9.7473068, -5.2320309, -9.7914791, -5.2038889, -4.0613270, 4.0763421
4: -10.9772940, -7.1570406, -11.0491438, -7.1286936, -3.5377893, 3.7032237
5: -0.2164798, 3.1693809, -0.2484035, 3.1760569, -3.1996918, 3.2484388
6: 4.5154767, 7.4828372, 4.4819517, 7.4914727, -2.6730618, 3.0008855
7: -18.0155163, -14.3241434, -18.0182629, -14.3083000, -3.1626987, 3.1552563
8: 0.1393671, 4.0413504, 0.1146286, 4.0746942, -3.6732502, 3.6665850
9: -8.8150845, -5.7976432, -8.8880892, -5.7797151, -2.6568670, 2.6905339

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4722925, upper bound: 1.4728598
time: 8.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733141, upper bound: 1.4728598
time: 5.13 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.5213165, -5.7260008, -9.5740128, -5.7006102, -3.2706585, 3.3169031
1: -13.1510210, -8.8311291, -13.2213202, -8.7191000, -3.7054873, 3.7184820
2: -8.1022339, -4.3769464, -8.1345129, -4.2285085, -3.8737254, 3.7575665
3: -9.7492638, -5.2276249, -9.8143997, -5.1187983, -4.1869011, 4.1124883
4: -10.9811182, -7.1514473, -11.1042881, -7.1095281, -3.5600481, 3.7645397
5: -0.2204189, 3.1744356, -0.3041427, 3.1941903, -3.2225056, 3.3200381
6: 4.5116248, 7.4887333, 4.4252977, 7.5128069, -2.6984935, 3.0634356
7: -18.0245075, -14.3196888, -18.0487118, -14.2578678, -3.2273483, 3.1879668
8: 0.1359887, 4.0514603, 0.0480578, 4.1131425, -3.7132645, 3.7237301
9: -8.8190060, -5.7852755, -8.9620895, -5.7470970, -2.6798849, 2.7549787

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4751836, upper bound: 1.4728597
time: 8.62 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4761981, upper bound: 1.4728595
time: 7.10 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.5391197, -5.7096634, -9.5346823, -5.7105403, -3.3971930, 3.2746735
1: -13.1964531, -8.7866535, -13.1846294, -8.7899265, -3.7016668, 3.6943772
2: -8.1096601, -4.3441486, -8.0927305, -4.3504071, -3.7592530, 3.7485819
3: -9.7992077, -5.1697454, -9.7975941, -5.1735482, -4.1463747, 4.1073728
4: -11.0640965, -7.0842457, -11.0601006, -7.0887566, -3.8246822, 3.7055016
5: -0.2583947, 3.1896379, -0.2551081, 3.1854413, -3.2544651, 3.2338405
6: 4.4682903, 7.5089517, 4.4715528, 7.5042710, -3.0359807, 3.0373988
7: -18.0321217, -14.2988205, -18.0248909, -14.3027725, -3.1852188, 3.1737356
8: 0.0909455, 4.0889635, 0.0937378, 4.0807028, -3.6957254, 3.7990193
9: -8.8969030, -5.7305169, -8.8935928, -5.7405286, -2.7473569, 2.6980801

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723589, upper bound: 1.4762210
time: 10.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733595, upper bound: 1.4762211
time: 5.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.5452461, -5.7085819, -9.5825977, -5.6961164, -3.4208546, 3.4391823
1: -13.2110710, -8.7825832, -13.2471199, -8.7146492, -3.7364869, 3.7563453
2: -8.1305208, -4.3364544, -8.1452084, -4.2182093, -3.9123116, 3.8087540
3: -9.8012562, -5.1651478, -9.8206425, -5.0881400, -4.2458544, 4.1447520
4: -11.0695314, -7.0786457, -11.1171780, -7.0696068, -3.8479166, 3.8507614
5: -0.2625117, 3.1949344, -0.3109789, 3.2038717, -3.2681341, 3.3092251
6: 4.4642563, 7.5148096, 4.4145732, 7.5255547, -3.0612984, 3.1002364
7: -18.0410957, -14.2939625, -18.0553265, -14.2519798, -3.2366133, 3.2077818
8: 0.0875138, 4.0993166, 0.0272765, 4.1195202, -3.8099270, 3.8451147
9: -8.9012518, -5.7181401, -8.9681673, -5.7078671, -2.7714210, 2.7645586

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4752214, upper bound: 1.4762214
time: 7.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762221, upper bound: 1.4762216
time: 5.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.10 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4722925, upper bound: 1.4728598
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4733141, upper bound: 1.4728598
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4751836, upper bound: 1.4728597
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4761981, upper bound: 1.4728595
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4723589, upper bound: 1.4762210
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4733595, upper bound: 1.4762211
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4752214, upper bound: 1.4762214
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.10
Output dim: 6, lower bound: -1.4762221, upper bound: 1.4762216

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.5076571, -5.7289171, -9.5239286, -5.7158241, -3.2400455, 3.2476859
1: -13.1311722, -8.8825855, -13.1566029, -8.8135986, -3.6150055, 3.5952468
2: -8.0711317, -4.3901858, -8.0778008, -4.3629532, -3.7081785, 3.6876149
3: -9.7167149, -5.2368178, -9.7789326, -5.2058043, -4.0258064, 4.0554218
4: -10.9711552, -7.1848302, -11.0465431, -7.1400633, -3.5208387, 3.6691637
5: -0.1746373, 3.1673374, -0.2312193, 3.1751943, -3.1565146, 3.2288938
6: 4.5300603, 7.4802608, 4.4880528, 7.4904404, -2.6553946, 2.9922080
7: -18.0129738, -14.3342400, -18.0172386, -14.3124838, -3.1552000, 3.1410155
8: 0.1461596, 4.0195103, 0.1173835, 4.0657334, -3.6474123, 3.6321292
9: -8.7891083, -5.8062515, -8.8774452, -5.7831073, -2.6240330, 2.6681032

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4728598
time: 5.47 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4728603
time: 4.87 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.5343246, -5.6910977, -9.5273848, -5.7150278, -3.2715406, 3.2928548
1: -13.2716503, -8.8213739, -13.1588116, -8.7940502, -3.7009163, 3.6552620
2: -8.1027308, -4.3623176, -8.0819874, -4.3604941, -3.7422366, 3.7196698
3: -9.7588568, -5.1422620, -9.7914600, -5.2038946, -4.0717955, 4.1520309
4: -11.0504417, -7.1437950, -11.0491381, -7.1287384, -3.5817156, 3.7155142
5: -0.2443314, 3.2749796, -0.2483559, 3.1760564, -3.2278404, 3.2965634
6: 4.4823399, 7.5202456, 4.4819756, 7.4914694, -2.7086248, 3.0382700
7: -18.0457592, -14.3135147, -18.0182590, -14.3083191, -3.1932478, 3.1648149
8: 0.0641164, 4.0565300, 0.1146356, 4.0746732, -3.7266035, 3.6759109
9: -8.8220673, -5.7220755, -8.8880672, -5.7797222, -2.6620426, 2.7414677

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4728622
time: 5.86 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4728603
time: 5.15 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5128269, -5.7279015, -9.5705490, -5.7014146, -3.2625217, 3.3115716
1: -13.1457367, -8.8787870, -13.2191343, -8.7386971, -3.6771626, 3.6687913
2: -8.0920000, -4.3829393, -8.1303177, -4.2309904, -3.8610096, 3.7473783
3: -9.7186718, -5.2324314, -9.8018589, -5.1207323, -4.1514320, 4.0915413
4: -10.9749794, -7.1792288, -11.1016893, -7.1208787, -3.5431108, 3.7304850
5: -0.1786020, 3.1723933, -0.2869339, 3.1933255, -3.1793113, 3.2983122
6: 4.5262089, 7.4861622, 4.4314404, 7.5117836, -2.6808319, 3.0547218
7: -18.0219669, -14.3297892, -18.0476952, -14.2620659, -3.2198443, 3.1737270
8: 0.1427976, 4.0296288, 0.0508237, 4.1041632, -3.6874084, 3.6887894
9: -8.7930117, -5.7938876, -8.9514189, -5.7504878, -2.6470814, 2.7318959

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718351, upper bound: 1.4728598
time: 5.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718351, upper bound: 1.4728602
time: 5.47 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5395908, -5.6900749, -9.5740023, -5.7006168, -3.2940488, 3.3567567
1: -13.2862167, -8.8175812, -13.2213182, -8.7191334, -3.7447948, 3.7288041
2: -8.1236067, -4.3550558, -8.1345024, -4.2285166, -3.8950901, 3.7794466
3: -9.7608070, -5.1378584, -9.8143787, -5.1188035, -4.1973782, 4.1814370
4: -11.0543261, -7.1381989, -11.1042805, -7.1095719, -3.6041527, 3.7768283
5: -0.2482862, 3.2800412, -0.3040972, 3.1941888, -3.2506461, 3.3513832
6: 4.4785056, 7.5261426, 4.4253206, 7.5128031, -2.7340398, 3.1008220
7: -18.0547638, -14.3090611, -18.0487061, -14.2578859, -3.2410221, 3.1975241
8: 0.0607393, 4.0666447, 0.0480655, 4.1131220, -3.7605467, 3.7330570
9: -8.8259716, -5.7096872, -8.9620667, -5.7471023, -2.6850510, 2.7922418

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4728611, upper bound: 1.4728599
time: 6.22 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4728611, upper bound: 1.4728595
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5306168, -5.7116690, -9.5312138, -5.7113457, -3.3881645, 3.2687349
1: -13.1910219, -8.8344460, -13.1824207, -8.8095007, -3.6759129, 3.6448369
2: -8.0994616, -4.3502340, -8.0885372, -4.3528786, -3.7465830, 3.7383032
3: -9.7686167, -5.1744838, -9.7850552, -5.1754599, -4.1110220, 4.0865397
4: -11.0574436, -7.1119666, -11.0574942, -7.1001177, -3.7994614, 3.6715007
5: -0.2165117, 3.1874652, -0.2379262, 3.1845751, -3.2112274, 3.2142916
6: 4.4831328, 7.5063963, 4.4776602, 7.5032382, -3.0201054, 3.0287361
7: -18.0295849, -14.3090124, -18.0238686, -14.3069611, -3.1777077, 3.1594563
8: 0.0977507, 4.0670843, 0.0964953, 4.0717440, -3.6698732, 3.7647095
9: -8.8708324, -5.7391887, -8.8829365, -5.7439241, -2.7158079, 2.6763721

Time for backsubstitution: 15.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4761975
time: 5.07 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4761983
time: 5.63 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5576591, -5.6734333, -9.5346708, -5.7105441, -3.4265480, 3.3132920
1: -13.3321819, -8.7731485, -13.1846266, -8.7899570, -3.7464809, 3.7053504
2: -8.1310720, -4.3219500, -8.0927181, -4.3504167, -3.7806554, 3.7707682
3: -9.8106880, -5.0799785, -9.7975731, -5.1735539, -4.1567860, 4.1755152
4: -11.1390991, -7.0709057, -11.0600920, -7.0887985, -3.8489189, 3.7179160
5: -0.2861896, 3.2956588, -0.2550631, 3.1854396, -3.2824988, 3.2863326
6: 4.4345903, 7.5463667, 4.4715776, 7.5042658, -3.0696754, 3.0747890
7: -18.0624504, -14.2879219, -18.0248833, -14.3027916, -3.2157021, 3.1835408
8: 0.0157728, 4.1041069, 0.0937458, 4.0806818, -3.7443037, 3.8084249
9: -8.9040470, -5.6547098, -8.8935719, -5.7405362, -2.7539206, 2.7494364

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4761975
time: 5.74 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699764, upper bound: 1.4761983
time: 5.46 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5367422, -5.7105870, -9.5791054, -5.6969213, -3.4118290, 3.4327612
1: -13.2056351, -8.8303804, -13.2449303, -8.7342501, -3.7082348, 3.7067809
2: -8.1203251, -4.3425593, -8.1410179, -4.2206998, -3.8996253, 3.7984586
3: -9.7706709, -5.1699023, -9.8081074, -5.0900722, -4.2103834, 4.1241922
4: -11.0628738, -7.1063595, -11.1145296, -7.0809488, -3.8227177, 3.8155899
5: -0.2206225, 3.1927619, -0.2937703, 3.2029972, -3.2249498, 3.2885213
6: 4.4790993, 7.5122576, 4.4207273, 7.5245328, -3.0454335, 3.0915303
7: -18.0385590, -14.3041611, -18.0543098, -14.2561808, -3.2286310, 3.1935010
8: 0.0943344, 4.0774417, 0.0300400, 4.1105604, -3.7849035, 3.8108292
9: -8.8751669, -5.7268119, -8.9574718, -5.7112622, -2.7398562, 2.7423761

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718340, upper bound: 1.4761975
time: 5.58 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718341, upper bound: 1.4761983
time: 5.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.5638914, -5.6723461, -9.5825863, -5.6961193, -3.4502063, 3.4785137
1: -13.3467884, -8.7690878, -13.2471123, -8.7146835, -3.7774286, 3.7672918
2: -8.1519423, -4.3142581, -8.1451979, -4.2182178, -3.9337244, 3.8309398
3: -9.8127308, -5.0753803, -9.8206215, -5.0881486, -4.2562728, 4.2062511
4: -11.1445789, -7.0653043, -11.1171684, -7.0696492, -3.8721342, 3.8651910
5: -0.2902904, 3.3009591, -0.3109319, 3.2038684, -3.2963190, 3.3426619
6: 4.4305696, 7.5522270, 4.4145966, 7.5255527, -3.0949831, 3.1376305
7: -18.0714321, -14.2830658, -18.0553265, -14.2519951, -3.2496176, 3.2175870
8: 0.0123390, 4.1144667, 0.0272843, 4.1195025, -3.8544335, 3.8545113
9: -8.9083767, -5.6423140, -8.9681463, -5.7078733, -2.7779751, 2.7991734

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4728601, upper bound: 1.4761974
time: 6.48 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4728601, upper bound: 1.4762217
time: 14.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 35.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4728598
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4728603
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4728622
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4728603
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4718351, upper bound: 1.4728598
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4718351, upper bound: 1.4728602
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4728611, upper bound: 1.4728599
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4728611, upper bound: 1.4728595
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4761975
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4761983
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4761975
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4699764, upper bound: 1.4761983
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4718340, upper bound: 1.4761975
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4718341, upper bound: 1.4761983
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4728601, upper bound: 1.4761974
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.99
Output dim: 6, lower bound: -1.4728601, upper bound: 1.4762217

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5076571, -5.7289171, -9.5086384, -5.7286105, -3.2334967, 3.2322407
1: -13.1311722, -8.8825855, -13.1224880, -8.8575783, -3.5693922, 3.5350049
2: -8.0711317, -4.3901858, -8.0602198, -4.3925934, -3.6785383, 3.6700339
3: -9.7167149, -5.2368178, -9.7332001, -5.2376084, -3.9910374, 4.0084705
4: -10.9711552, -7.1848302, -10.9717207, -7.1729484, -3.4881248, 3.4764252
5: -0.1746373, 3.1673374, -0.1962614, 3.1644776, -3.1674557, 3.1920567
6: 4.5300603, 7.4802608, 4.5246100, 7.4770713, -2.6430125, 2.6526566
7: -18.0129738, -14.3342400, -18.0072346, -14.3319740, -3.1372480, 3.1259766
8: 0.1461596, 4.0195103, 0.1448692, 4.0242243, -3.6047564, 3.6035647
9: -8.7891083, -5.8062515, -8.8012218, -5.8110352, -2.5672169, 2.5867057

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4699751
time: 5.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4728599
time: 5.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5076571, -5.7289171, -9.5311546, -5.7113523, -3.2423854, 3.2546492
1: -13.1311722, -8.8825855, -13.1823034, -8.8095779, -3.6073322, 3.6134071
2: -8.0711317, -4.3901858, -8.0881186, -4.3528957, -3.7182360, 3.6979327
3: -9.7167149, -5.2368178, -9.7850494, -5.1756997, -4.0566692, 4.0620995
4: -10.9711552, -7.1848302, -11.0574226, -7.1001339, -3.5429096, 3.6611538
5: -0.1746373, 3.1673374, -0.2378745, 3.1840577, -3.1654549, 3.2367125
6: 4.5300603, 7.4802608, 4.4795194, 7.5032320, -2.6701202, 3.0007415
7: -18.0129738, -14.3342400, -18.0237846, -14.3069725, -3.1606703, 3.1433592
8: 0.1461596, 4.0195103, 0.0965495, 4.0717278, -3.6455240, 3.6502676
9: -8.7891083, -5.8062515, -8.8829145, -5.7439394, -2.6408877, 2.6322451

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4699755
time: 5.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689449, upper bound: 1.4728602
time: 5.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.5343246, -5.6910977, -9.5121193, -5.7278566, -3.2646875, 3.2774158
1: -13.2716503, -8.8213739, -13.1246290, -8.8380852, -3.6552286, 3.5938883
2: -8.1027308, -4.3623176, -8.0644178, -4.3901739, -3.7125568, 3.7021003
3: -9.7588568, -5.1422620, -9.7457275, -5.2356820, -4.0370560, 4.1049571
4: -11.0504417, -7.1437950, -10.9741459, -7.1615992, -3.5582190, 3.5183616
5: -0.2443314, 3.2749796, -0.2133703, 3.1652923, -3.2396860, 3.2596755
6: 4.4823399, 7.5202456, 4.5186439, 7.4781084, -2.6962318, 2.7057436
7: -18.0457592, -14.3135147, -18.0082569, -14.3278408, -3.1753273, 3.1497746
8: 0.0641164, 4.0565300, 0.1421256, 4.0331578, -3.6839209, 3.6473408
9: -8.8220673, -5.7220755, -8.8118315, -5.8076725, -2.6055508, 2.6601286

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4699750
time: 6.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4728599
time: 5.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.5343246, -5.6910977, -9.5346098, -5.7105503, -3.2738390, 3.2998118
1: -13.2716503, -8.8213739, -13.1845055, -8.7900352, -3.6749544, 3.6734099
2: -8.1027308, -4.3623176, -8.0923033, -4.3504343, -3.7522964, 3.7299857
3: -9.7588568, -5.1422620, -9.7975721, -5.1737962, -4.1026516, 4.1259961
4: -11.0504417, -7.1437950, -11.0600166, -7.0888152, -3.5846958, 3.7075000
5: -0.2443314, 3.2749796, -0.2550101, 3.1849208, -3.2367802, 3.2951965
6: 4.4823399, 7.5202456, 4.4734354, 7.5042601, -2.7233453, 3.0468102
7: -18.0457592, -14.3135147, -18.0248032, -14.3027992, -3.1987314, 3.1671543
8: 0.0641164, 4.0565300, 0.0938008, 4.0806670, -3.7074332, 3.6945572
9: -8.8220673, -5.7220755, -8.8935518, -5.7405500, -2.6790242, 2.6925790

Time for backsubstitution: 15.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4699756
time: 5.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699777, upper bound: 1.4728603
time: 5.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5128269, -5.7279015, -9.5552435, -5.7144747, -3.2472763, 3.2960773
1: -13.1457367, -8.8787870, -13.1850309, -8.7828579, -3.6314955, 3.6252651
2: -8.0920000, -4.3829393, -8.1131563, -4.2608891, -3.8311110, 3.7302170
3: -9.7186718, -5.2324314, -9.7561073, -5.1531243, -4.1162987, 4.0444479
4: -10.9749794, -7.1792288, -11.0258932, -7.1536198, -3.5105219, 3.6550884
5: -0.1786020, 3.1723933, -0.2517419, 3.1825976, -3.1679602, 3.2615318
6: 4.5262089, 7.4861622, 4.4687157, 7.4988575, -2.6688924, 3.0174465
7: -18.0219669, -14.3297892, -18.0377598, -14.2818460, -3.1976986, 3.1585798
8: 0.1427976, 4.0296288, 0.0783236, 4.0627875, -3.6447220, 3.6644454
9: -8.7930117, -5.7938876, -8.8745775, -5.7786837, -2.6200752, 2.6500304

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4699753
time: 5.23 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4728601
time: 4.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5128269, -5.7279015, -9.5778255, -5.6969290, -3.2649727, 3.3185835
1: -13.1457367, -8.8787870, -13.2448101, -8.7346096, -3.6512823, 3.6759419
2: -8.0920000, -4.3829393, -8.1406021, -4.2209053, -3.8710947, 3.7576628
3: -9.7186718, -5.2324314, -9.8079748, -5.0903134, -4.1585908, 4.0982294
4: -10.9749794, -7.1792288, -11.1126108, -7.0809622, -3.5653892, 3.7037501
5: -0.1786020, 3.1723933, -0.2937174, 3.2021747, -3.1882448, 3.2967124
6: 4.5262089, 7.4861622, 4.4225855, 7.5244927, -2.6953893, 3.0635767
7: -18.0219669, -14.3297892, -18.0542259, -14.2564106, -3.2212110, 3.1760244
8: 0.1427976, 4.0296288, 0.0300965, 4.1100912, -3.6794276, 3.6944432
9: -8.7930117, -5.7938876, -8.9568930, -5.7112784, -2.6630216, 2.6826649

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4699755
time: 5.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4728602
time: 5.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5395908, -5.6900749, -9.5587158, -5.7136765, -3.2787991, 3.3412704
1: -13.2862167, -8.8175812, -13.1872168, -8.7633104, -3.6991267, 3.6852751
2: -8.1236067, -4.3550558, -8.1173506, -4.2584085, -3.8651981, 3.7622948
3: -9.7608070, -5.1378584, -9.7686281, -5.1511769, -4.1622610, 4.1343770
4: -11.0543261, -7.1381989, -11.0284882, -7.1422915, -3.5807695, 3.7014303
5: -0.2482862, 3.2800412, -0.2688775, 3.1834559, -3.2392869, 3.3145339
6: 4.4785056, 7.5261426, 4.4626141, 7.4998841, -2.7221041, 3.0635285
7: -18.0547638, -14.3090611, -18.0387745, -14.2776756, -3.2188864, 3.1823745
8: 0.0607393, 4.0666447, 0.0755711, 4.0717115, -3.7179828, 3.7087131
9: -8.8259716, -5.7096872, -8.8852139, -5.7752991, -2.6580658, 2.7103653

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4699753
time: 5.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4728600
time: 4.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.5395908, -5.6900749, -9.5812798, -5.6961250, -3.2964993, 3.3637619
1: -13.2862167, -8.8175812, -13.2469959, -8.7150526, -3.7189121, 3.7361372
2: -8.1236067, -4.3550558, -8.1447821, -4.2184286, -3.9051781, 3.7897263
3: -9.7608070, -5.1378584, -9.8204861, -5.0883880, -4.2045298, 4.1554704
4: -11.0543261, -7.1381989, -11.1152086, -7.0696645, -3.6071634, 3.7500806
5: -0.2482862, 3.2800412, -0.3108795, 3.2030382, -3.2595792, 3.3496962
6: 4.4785056, 7.5261426, 4.4164534, 7.5255098, -2.7485929, 3.1010714
7: -18.0547638, -14.3090611, -18.0552425, -14.2522278, -3.2423930, 3.1998177
8: 0.0607393, 4.0666447, 0.0273417, 4.1190238, -3.7413292, 3.7387242
9: -8.8259716, -5.7096872, -8.9675522, -5.7078876, -2.7011604, 2.7429886

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4699752
time: 5.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4728604
time: 4.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5305586, -5.7116737, -9.5086384, -5.7286105, -3.3541217, 3.2413607
1: -13.1909113, -8.8345213, -13.1224880, -8.8575783, -3.6443338, 3.5628521
2: -8.0990496, -4.3502488, -8.0602198, -4.3925934, -3.7064562, 3.7099710
3: -9.7686167, -5.1747122, -9.7332001, -5.2376084, -4.0447645, 4.0743179
4: -11.0573721, -7.1119676, -10.9717207, -7.1729484, -3.7257662, 3.5321088
5: -0.2164614, 3.1869526, -0.1962614, 3.1644776, -3.2122211, 3.1822371
6: 4.4849887, 7.5063949, 4.5246100, 7.4770713, -2.9920826, 2.6792245
7: -18.0295048, -14.3090229, -18.0072346, -14.3319740, -3.1546211, 3.1419473
8: 0.0977985, 4.0670700, 0.1448692, 4.0242243, -3.6522846, 3.7140574
9: -8.8708143, -5.7391887, -8.8012218, -5.8110352, -2.5904908, 2.6559002

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4733124
time: 5.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689458, upper bound: 1.4761975
time: 4.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5306168, -5.7116690, -9.5312128, -5.7113476, -3.3881621, 3.2687330
1: -13.1910219, -8.8344460, -13.1824121, -8.8095036, -3.6759105, 3.6251450
2: -8.0994616, -4.3502340, -8.0885315, -4.3528819, -3.7465796, 3.7382975
3: -9.7686167, -5.1744838, -9.7850523, -5.1754708, -4.0700207, 4.0865369
4: -11.0574436, -7.1119666, -11.0574894, -7.1001301, -3.7669201, 3.6714983
5: -0.2165117, 3.1874652, -0.2379246, 3.1845725, -3.1956048, 3.2142901
6: 4.4831328, 7.5063963, 4.4776640, 7.5032334, -3.0201006, 3.0287323
7: -18.0295849, -14.3090124, -18.0238609, -14.3069630, -3.1777058, 3.1543889
8: 0.0977507, 4.0670843, 0.0965022, 4.0717416, -3.6698704, 3.7436056
9: -8.8708324, -5.7391887, -8.8829346, -5.7439384, -2.6601572, 2.6763697

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4733132
time: 5.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689437, upper bound: 1.4762223
time: 5.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.5576029, -5.6734362, -9.5121193, -5.7278566, -3.3920350, 3.2859302
1: -13.3320684, -8.7732220, -13.1246290, -8.8380852, -3.7136049, 3.6218209
2: -8.1306648, -4.3219657, -8.0644178, -4.3901739, -3.7404909, 3.7424521
3: -9.8106861, -5.0802078, -9.7457275, -5.2356820, -4.0905552, 4.1412067
4: -11.1390276, -7.0709095, -10.9741459, -7.1615992, -3.7752514, 3.5741887
5: -0.2861385, 3.2951441, -0.2133703, 3.1652923, -3.2843695, 3.2485132
6: 4.4364471, 7.5463667, 4.5186439, 7.4781084, -3.0416613, 2.7213223
7: -18.0623665, -14.2879314, -18.0082569, -14.3278408, -3.1927872, 3.1660342
8: 0.0158212, 4.1040940, 0.1421256, 4.0331578, -3.7142663, 3.7577715
9: -8.9040298, -5.6547132, -8.8118315, -5.8076725, -2.6294527, 2.7157142

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4733124
time: 5.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4761976
time: 5.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.5576591, -5.6734333, -9.5346689, -5.7105465, -3.4265461, 3.3132887
1: -13.3321819, -8.7731485, -13.1846180, -8.7899590, -3.7464786, 3.6856022
2: -8.1310720, -4.3219500, -8.0927143, -4.3504190, -3.7806530, 3.7707644
3: -9.8106880, -5.0799785, -9.7975750, -5.1735668, -4.1157837, 4.1727538
4: -11.1390991, -7.0709057, -11.0600882, -7.0888109, -3.8237076, 3.7179136
5: -0.2861896, 3.2956588, -0.2550614, 3.1854353, -3.2668705, 3.2863307
6: 4.4345903, 7.5463667, 4.4715791, 7.5042629, -3.0696726, 3.0747876
7: -18.0624504, -14.2879219, -18.0248833, -14.3027916, -3.2157011, 3.1784329
8: 0.0157728, 4.1041069, 0.0937506, 4.0806813, -3.7424755, 3.7867622
9: -8.9040470, -5.6547098, -8.8935699, -5.7405481, -2.6982112, 2.7489202

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699766, upper bound: 1.4733130
time: 5.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699764, upper bound: 1.4762223
time: 5.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5366840, -5.7105923, -9.5552435, -5.7144747, -3.3812280, 3.2984781
1: -13.2055206, -8.8304539, -13.1850309, -8.7828579, -3.6878858, 3.6449552
2: -8.1199121, -4.3425746, -8.1131563, -4.2608891, -3.8590231, 3.7705817
3: -9.7706690, -5.1701326, -9.7561073, -5.1531243, -4.1443181, 4.1104779
4: -11.0628023, -7.1063633, -11.0258932, -7.1536198, -3.7492313, 3.7026172
5: -0.2205732, 3.1922479, -0.2517419, 3.1825976, -3.2133684, 3.2512517
6: 4.4809585, 7.5122566, 4.4687157, 7.4988575, -3.0178990, 3.0435410
7: -18.0384789, -14.3041687, -18.0377598, -14.2818460, -3.2125874, 3.1760607
8: 0.0943797, 4.0774269, 0.0783236, 4.0627875, -3.6864400, 3.7606750
9: -8.8751488, -5.7268152, -8.8745775, -5.7786837, -2.6724472, 2.7048554

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689426, upper bound: 1.4733124
time: 5.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689426, upper bound: 1.4761998
time: 4.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5367422, -5.7105870, -9.5791016, -5.6969223, -3.4118252, 3.4420338
1: -13.2056351, -8.8303804, -13.2449236, -8.7342510, -3.7082329, 3.6969373
2: -8.1203251, -4.3425593, -8.1410151, -4.2207036, -3.8996215, 3.7984557
3: -9.7706709, -5.1699023, -9.8081045, -5.0900850, -4.1912870, 4.1241899
4: -11.0628738, -7.1063595, -11.1145248, -7.0809612, -3.7901001, 3.8081007
5: -0.2206225, 3.1927619, -0.2937665, 3.2029943, -3.2113080, 3.2885191
6: 4.4790993, 7.5122576, 4.4207296, 7.5245275, -3.0454283, 3.0915279
7: -18.0385590, -14.3041611, -18.0543060, -14.2561827, -3.2285652, 3.1885343
8: 0.0943344, 4.0774417, 0.0300448, 4.1105585, -3.7849016, 3.7989001
9: -8.8751669, -5.7268119, -8.9574699, -5.7112751, -2.6886926, 2.7396772

Time for backsubstitution: 14.58 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.050609588623047
rel_dist={6: [-1.4762376626074394, 1.476239686090869]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041193, upper bound: 1.2029565
time: 5.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041213, upper bound: 1.2041183
time: 5.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.42
Output dim: 6, lower bound: -1.2041193, upper bound: 1.2029565
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.42
Output dim: 6, lower bound: -1.2041213, upper bound: 1.2041183

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.5213375, -5.7259974, -9.5337420, -5.7148738, -3.1013823, 3.1084886
1: -13.1510839, -8.8311186, -13.1750145, -8.7886448, -3.4578395, 3.4741325
2: -8.1023321, -4.3769217, -8.1155434, -4.3509707, -3.6916966, 3.6551313
3: -9.7492733, -5.2276101, -9.7925758, -5.2078443, -3.8549728, 3.8767519
4: -10.9811316, -7.1514211, -11.0523043, -7.1346302, -3.3298330, 3.4971805
5: -0.2204313, 3.1744542, -0.2530255, 3.1815286, -3.0970135, 3.1462574
6: 4.5116096, 7.4887538, 4.4789152, 7.4968810, -2.5569692, 2.9166722
7: -18.0245476, -14.3196735, -18.0318699, -14.3021412, -2.9666510, 2.9622536
8: 0.1359761, 4.0514793, 0.1167593, 4.0904627, -3.5763826, 3.5561399
9: -8.8190193, -5.7852249, -8.8933535, -5.7729669, -2.5078096, 2.5448194

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032536, upper bound: 1.2029510
time: 5.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041143, upper bound: 1.2029522
time: 5.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.5452709, -5.7085772, -9.5452719, -5.7085762, -3.2456608, 3.2379260
1: -13.2111359, -8.7825670, -13.2111368, -8.7825718, -3.5025349, 3.5258834
2: -8.1306238, -4.3364310, -8.1306238, -4.3364277, -3.7213812, 3.7343497
3: -9.8012638, -5.1651335, -9.8012657, -5.1651297, -3.9524240, 3.9075251
4: -11.0695486, -7.0786190, -11.0695524, -7.0786142, -3.6485214, 3.6141992
5: -0.2625251, 3.1949496, -0.2625258, 3.1949520, -3.1479745, 3.1325936
6: 4.4642420, 7.5148306, 4.4642410, 7.5148335, -2.9765887, 2.9431310
7: -18.0411339, -14.2939463, -18.0411358, -14.2939463, -2.9790325, 2.9849849
8: 0.0875016, 4.0993347, 0.0875000, 4.0993357, -3.6788092, 3.7003818
9: -8.9012642, -5.7180891, -8.9012661, -5.7180829, -2.6258616, 2.5643725

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032536, upper bound: 1.2041130
time: 5.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041143, upper bound: 1.2041135
time: 5.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.65 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 25.65
Output dim: 6, lower bound: -1.2032536, upper bound: 1.2029510
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.65
Output dim: 6, lower bound: -1.2041143, upper bound: 1.2029522
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.65
Output dim: 6, lower bound: -1.2032536, upper bound: 1.2041130
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.65
Output dim: 6, lower bound: -1.2041143, upper bound: 1.2041135

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.5213079, -5.7260017, -9.5709391, -5.7027240, -3.1119251, 3.1524067
1: -13.1509943, -8.8311357, -13.2097692, -8.7209997, -3.4964833, 3.5060010
2: -8.1021938, -4.3769565, -8.1301041, -4.2347360, -3.7639894, 3.6552668
3: -9.7492619, -5.2276316, -9.8115435, -5.1351643, -3.9489517, 3.8921118
4: -10.9811134, -7.1514568, -11.0992804, -7.1257958, -3.3370018, 3.5437317
5: -0.2204139, 3.1744313, -0.3009419, 3.1903036, -3.1051898, 3.2023919
6: 4.5116305, 7.4887242, 4.4310441, 7.5077076, -2.5656343, 2.9630184
7: -18.0244904, -14.3196945, -18.0456753, -14.2603207, -3.0088968, 2.9749203
8: 0.1359934, 4.0514545, 0.0567617, 4.1102204, -3.5878105, 3.5938582
9: -8.8190031, -5.7852950, -8.9598789, -5.7635798, -2.5010014, 2.5884950

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037724, upper bound: 1.2029517
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041138, upper bound: 1.2029516
time: 5.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.5361938, -5.7102141, -9.5346813, -5.7105436, -3.1134815, 3.1165414
1: -13.1890526, -8.7887306, -13.1846266, -8.7899246, -3.4841461, 3.4971857
2: -8.0990696, -4.3480835, -8.0927267, -4.3504095, -3.6822829, 3.6893396
3: -9.7981768, -5.1721220, -9.7975922, -5.1735549, -3.9264650, 3.8815928
4: -11.0613918, -7.0870857, -11.0600977, -7.0887623, -3.5471802, 3.4780469
5: -0.2563066, 3.1869738, -0.2551081, 3.1854401, -3.1397448, 3.1228991
6: 4.4703336, 7.5060024, 4.4715533, 7.5042667, -2.9612875, 2.9277158
7: -18.0275841, -14.3013134, -18.0248871, -14.3027735, -2.9698496, 2.9700689
8: 0.0926975, 4.0837469, 0.0937415, 4.0807028, -3.5725813, 3.6008568
9: -8.8947744, -5.7367883, -8.8935938, -5.7405367, -2.5899682, 2.5247822

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2029087, upper bound: 1.2041122
time: 5.12 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032532, upper bound: 1.2041132
time: 4.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.5452356, -5.7085848, -9.5825825, -5.6961460, -3.2570033, 3.2766023
1: -13.2110462, -8.7825890, -13.2469807, -8.7146740, -3.5259333, 3.5463707
2: -8.1304817, -4.3364635, -8.1451912, -4.2184348, -3.7966123, 3.7347841
3: -9.8012533, -5.1651545, -9.8206043, -5.0885816, -4.0202422, 3.9233394
4: -11.0695229, -7.0786543, -11.1171227, -7.0696368, -3.6555614, 3.6488848
5: -0.2625067, 3.1949284, -0.3109326, 3.2038538, -3.1560221, 3.1949282
6: 4.4642630, 7.5148015, 4.4147196, 7.5255518, -2.9874721, 2.9893751
7: -18.0410805, -14.2939720, -18.0552864, -14.2519932, -3.0201559, 2.9982796
8: 0.0875171, 4.0993090, 0.0273169, 4.1194544, -3.6896858, 3.7266517
9: -8.9012451, -5.7181582, -8.9681644, -5.7079535, -2.6191630, 2.6065884

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037724, upper bound: 1.2041132
time: 5.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041138, upper bound: 1.2041131
time: 5.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.21 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 6, lower bound: -1.2037724, upper bound: 1.2029517
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 6, lower bound: -1.2041138, upper bound: 1.2029516
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 6, lower bound: -1.2029087, upper bound: 1.2041122
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 6, lower bound: -1.2032532, upper bound: 1.2041132
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 6, lower bound: -1.2037724, upper bound: 1.2041132
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 6, lower bound: -1.2041138, upper bound: 1.2041131

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5128183, -5.7279015, -9.5643120, -5.7042737, -3.1025915, 3.1446295
1: -13.1457119, -8.8787909, -13.2056332, -8.7584381, -3.4513164, 3.4539094
2: -8.0919590, -4.3829517, -8.1221008, -4.2394934, -3.7267189, 3.6199908
3: -9.7186661, -5.2324381, -9.7875872, -5.1388907, -3.9105830, 3.8590779
4: -10.9749756, -7.1792388, -11.0942373, -7.1474886, -3.3096581, 3.5058265
5: -0.1785965, 3.1723866, -0.2680840, 3.1886311, -3.0611062, 3.1656227
6: 4.5262165, 7.4861536, 4.4427638, 7.5057344, -2.5470457, 2.9447956
7: -18.0219555, -14.3297958, -18.0437202, -14.2683201, -2.9960370, 2.9599557
8: 0.1428028, 4.0296216, 0.0620800, 4.0930662, -3.5512161, 3.5538516
9: -8.7930040, -5.7939062, -8.9394932, -5.7702398, -2.4644718, 2.5547757

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2026267
time: 5.49 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2029510
time: 5.79 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5376539, -5.6929932, -9.5709190, -5.7027287, -3.1318979, 3.1884985
1: -13.2735643, -8.8191204, -13.2097578, -8.7210503, -3.5263004, 3.5117526
2: -8.1218491, -4.3579435, -8.1300888, -4.2347479, -3.7801318, 3.6672988
3: -9.7599516, -5.1489286, -9.8115149, -5.1351728, -3.9584656, 3.9464808
4: -11.0473833, -7.1393209, -11.0992689, -7.1258578, -3.3750358, 3.5531898
5: -0.2443762, 3.2710776, -0.3008757, 3.1903000, -3.1277790, 3.2266247
6: 4.4825525, 7.5251799, 4.4310784, 7.5077047, -2.5966592, 2.9857101
7: -18.0531082, -14.3103962, -18.0456715, -14.2603445, -3.0199318, 2.9829926
8: 0.0695181, 4.0656662, 0.0567703, 4.1101923, -3.6209006, 3.6017241
9: -8.8254757, -5.7185564, -8.9598446, -5.7635884, -2.5055289, 2.6168189

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2026269
time: 5.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2029513
time: 5.95 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5277624, -5.7122178, -9.5280514, -5.7121010, -3.1042719, 3.1080589
1: -13.1836300, -8.8365040, -13.1803560, -8.8273335, -3.4408054, 3.4450643
2: -8.0888700, -4.3541498, -8.0847301, -4.3551478, -3.6451402, 3.6515579
3: -9.7675953, -5.1768498, -9.7736406, -5.1772375, -3.8881264, 3.8486924
4: -11.0548420, -7.1148109, -11.0550261, -7.1104774, -3.5125065, 3.4401836
5: -0.2144091, 3.1848190, -0.2222795, 3.1837626, -3.0956287, 3.0874448
6: 4.4851813, 7.5034494, 4.4832020, 7.5022802, -2.9392881, 2.9095097
7: -18.0250511, -14.3114929, -18.0229111, -14.3107586, -2.9572945, 2.9549837
8: 0.0994968, 4.0618877, 0.0990429, 4.0635834, -3.5360031, 3.5608096
9: -8.8687420, -5.7454567, -8.8732204, -5.7472115, -2.5534124, 2.4922719

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2037855
time: 5.36 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2029084, upper bound: 1.2041118
time: 5.17 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5524578, -5.6769381, -9.5346622, -5.7105484, -3.1333575, 3.1513720
1: -13.3120928, -8.7768021, -13.1846180, -8.7899742, -3.5176449, 3.5031500
2: -8.1187458, -4.3288517, -8.0927105, -4.3504210, -3.6980004, 3.7007265
3: -9.8087902, -5.0934219, -9.7975636, -5.1735630, -3.9359484, 3.9338021
4: -11.1288729, -7.0748625, -11.0600872, -7.0888262, -3.5636530, 3.4876213
5: -0.2802060, 3.2839253, -0.2550402, 3.1854355, -3.1622448, 3.1670442
6: 4.4407468, 7.5424552, 4.4715881, 7.5042629, -2.9928083, 2.9557457
7: -18.0562611, -14.2918215, -18.0248852, -14.3027992, -2.9977398, 2.9785423
8: 0.0262935, 4.0978312, 0.0937525, 4.0806732, -3.6056118, 3.6087108
9: -8.9013023, -5.6698608, -8.8935614, -5.7405462, -2.5945978, 2.5659506

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032527, upper bound: 1.2037858
time: 5.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032529, upper bound: 1.2041122
time: 5.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5367336, -5.7105875, -9.5759048, -5.6977043, -3.2467470, 3.2673192
1: -13.2056036, -8.8303852, -13.2428064, -8.7521219, -3.4811630, 3.4945080
2: -8.1202869, -4.3425694, -8.1372061, -4.2232041, -3.7593112, 3.6970015
3: -9.7706661, -5.1699080, -9.7966566, -5.0923014, -3.9819908, 3.8907189
4: -11.0628672, -7.1063704, -11.1119719, -7.0913076, -3.6194859, 3.6096940
5: -0.2206159, 3.1927562, -0.2780533, 3.2021618, -3.1119466, 3.1583896
6: 4.4791064, 7.5122480, 4.4264579, 7.5235810, -2.9654331, 2.9708195
7: -18.0385456, -14.3041649, -18.0533199, -14.2600088, -3.0073643, 2.9832730
8: 0.0943373, 4.0774345, 0.0326316, 4.1023359, -3.6540623, 3.6875100
9: -8.8751621, -5.7268305, -8.9477215, -5.7146254, -2.5841351, 2.5746045

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2037865
time: 5.71 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2041157
time: 5.99 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.5619087, -5.6752992, -9.5825644, -5.6961479, -3.2817121, 3.3120947
1: -13.3340702, -8.7706223, -13.2469711, -8.7147217, -3.5569553, 3.5526547
2: -8.1501741, -4.3172011, -8.1451759, -4.2184453, -3.8129425, 3.7463083
3: -9.8118811, -5.0864558, -9.8205738, -5.0885901, -4.0298109, 3.9683661
4: -11.1374187, -7.0664320, -11.1171141, -7.0696979, -3.6746120, 3.6608076
5: -0.2864006, 3.2919428, -0.3108644, 3.2038503, -3.1787081, 3.2201695
6: 4.4346809, 7.5512590, 4.4147549, 7.5255489, -3.0189342, 2.9988151
7: -18.0697689, -14.2844315, -18.0552788, -14.2520170, -3.0310493, 3.0065494
8: 0.0211126, 4.1134839, 0.0273283, 4.1194263, -3.7182660, 3.7346196
9: -8.9078598, -5.6512051, -8.9681320, -5.7079620, -2.6250310, 2.6336832

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2037865
time: 5.41 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2041128
time: 5.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.07 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2026267
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2029510
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2026269
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2029513
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2037855
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2029084, upper bound: 1.2041118
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2032527, upper bound: 1.2037858
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2032529, upper bound: 1.2041122
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2037865
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2041157
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2037865
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.07
Output dim: 6, lower bound: -1.2041136, upper bound: 1.2041128

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5092010, -5.7300420, -9.5486746, -5.7123528, -3.0887971, 3.1265807
1: -13.1414709, -8.8940811, -13.1813278, -8.7807980, -3.4267287, 3.4114799
2: -8.0886889, -4.3879395, -8.1080656, -4.2523828, -3.7100792, 3.5999737
3: -9.7156782, -5.2361517, -9.7719736, -5.1505232, -3.8929024, 3.8372164
4: -10.9689865, -7.1886454, -11.0718174, -7.1685743, -3.2816372, 3.4825983
5: -0.1709437, 3.1704359, -0.2517118, 3.1801648, -3.0448098, 3.1470230
6: 4.5377359, 7.4852448, 4.4637756, 7.4973488, -2.5266953, 2.9216042
7: -18.0178719, -14.3530817, -18.0202808, -14.3065176, -2.9526567, 2.9130363
8: 0.1473749, 4.0264769, 0.0778494, 4.0815449, -3.5333548, 3.5274549
9: -8.7884769, -5.8040895, -8.9269876, -5.7894983, -2.4370995, 2.5305843

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037740, upper bound: 1.2022262
time: 5.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037718, upper bound: 1.2026264
time: 5.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5128155, -5.7279053, -9.5643110, -5.7042761, -3.1022348, 3.1424475
1: -13.1457090, -8.8787975, -13.2056322, -8.7584448, -3.4315510, 3.4507449
2: -8.0919580, -4.3829508, -8.1220999, -4.2394953, -3.7239189, 3.6195946
3: -9.7186689, -5.2324390, -9.7875843, -5.1388931, -3.9030924, 3.8572369
4: -10.9749737, -7.1792431, -11.0942354, -7.1474919, -3.3078890, 3.5113535
5: -0.1785927, 3.1723859, -0.2680807, 3.1886303, -3.0619478, 3.1626124
6: 4.5262203, 7.4861526, 4.4427681, 7.5057349, -2.5470428, 2.9398174
7: -18.0219517, -14.3298035, -18.0437164, -14.2683277, -2.9756403, 2.9599504
8: 0.1428034, 4.0296202, 0.0620819, 4.0930634, -3.5474844, 3.5455747
9: -8.7930040, -5.7939091, -8.9394913, -5.7702422, -2.4628181, 2.5513892

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2025557
time: 5.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2029510
time: 5.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5340033, -5.6951275, -9.5552702, -5.7108030, -3.1180730, 3.1704693
1: -13.2693300, -8.8344097, -13.1854591, -8.7434006, -3.5016422, 3.4693303
2: -8.1185074, -4.3629341, -8.1160526, -4.2476377, -3.7634268, 3.6472869
3: -9.7569513, -5.1526527, -9.7959080, -5.1468096, -3.9407482, 3.9231801
4: -11.0413647, -7.1487174, -11.0768433, -7.1469383, -3.3477492, 3.5299664
5: -0.2367353, 3.2691224, -0.2845120, 3.1818340, -3.1115413, 3.2080212
6: 4.4940896, 7.5242710, 4.4521112, 7.4993196, -2.5763049, 2.9626164
7: -18.0490303, -14.3336735, -18.0222340, -14.2985382, -2.9765606, 2.9361095
8: 0.0741001, 4.0624838, 0.0725405, 4.0986795, -3.6027451, 3.5752797
9: -8.8209343, -5.7287407, -8.9473524, -5.7828522, -2.4781566, 2.5926304

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2022262
time: 5.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2026266
time: 5.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.5376520, -5.6929936, -9.5709209, -5.7027278, -3.1314621, 3.1859512
1: -13.2735596, -8.8191223, -13.2097588, -8.7210569, -3.5065060, 3.5087149
2: -8.1218500, -4.3579445, -8.1300869, -4.2347507, -3.7772961, 3.6669078
3: -9.7599535, -5.1489296, -9.8115158, -5.1351738, -3.9509573, 3.9393144
4: -11.0473824, -7.1393223, -11.0992680, -7.1258602, -3.3679314, 3.5587144
5: -0.2443728, 3.2710776, -0.3008709, 3.1903000, -3.1286244, 3.2236063
6: 4.4825554, 7.5251789, 4.4310832, 7.5077047, -2.5966563, 2.9774828
7: -18.0531101, -14.3104029, -18.0456734, -14.2603512, -2.9995499, 2.9829869
8: 0.0695176, 4.0656648, 0.0567706, 4.1101904, -3.6119890, 3.5934014
9: -8.8254728, -5.7185578, -8.9598446, -5.7635913, -2.5038767, 2.6134374

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2025558
time: 5.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2029510
time: 5.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5241423, -5.7146950, -9.5124226, -5.7201567, -3.0904841, 3.0897479
1: -13.1787682, -8.8518353, -13.1560402, -8.8496752, -3.4136128, 3.4019623
2: -8.0855541, -4.3595376, -8.0707684, -4.3680010, -3.6288495, 3.6309924
3: -9.7645760, -5.1806154, -9.7580681, -5.1887374, -3.8707981, 3.8268032
4: -11.0474672, -7.1242065, -11.0326548, -7.1315556, -3.4853821, 3.4170923
5: -0.2067518, 3.1825106, -0.2059362, 3.1752872, -3.0794187, 3.0686703
6: 4.4974666, 7.5025373, 4.5041285, 7.4938841, -2.9184322, 2.8863215
7: -18.0209713, -14.3349667, -17.9993935, -14.3489714, -2.9140768, 2.9078732
8: 0.1041126, 4.0587215, 0.1147404, 4.0521059, -3.5181074, 3.5344567
9: -8.8641634, -5.7558198, -8.8606577, -5.7664642, -2.5260968, 2.4679494

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2029104
time: 5.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2037855
time: 5.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5277643, -5.7122183, -9.5280504, -5.7120991, -3.1038942, 3.1049500
1: -13.1836281, -8.8365097, -13.1803532, -8.8273363, -3.4221969, 3.4357593
2: -8.0888681, -4.3541508, -8.0847282, -4.3551483, -3.6438885, 3.6509404
3: -9.7675972, -5.1768494, -9.7736397, -5.1772389, -3.8843975, 3.8467817
4: -11.0548401, -7.1148124, -11.0550270, -7.1104808, -3.5037823, 3.4457088
5: -0.2144072, 3.1848190, -0.2222767, 3.1837630, -3.0964723, 3.0869656
6: 4.4851856, 7.5034485, 4.4832072, 7.5022783, -2.9359155, 2.9045320
7: -18.0250435, -14.3115005, -18.0229130, -14.3107672, -2.9447279, 2.9549785
8: 0.0994973, 4.0618892, 0.0990429, 4.0635824, -3.5322485, 3.5525641
9: -8.8687429, -5.7454605, -8.8732185, -5.7472138, -2.5492783, 2.4922690

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2037182
time: 5.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2041116
time: 5.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.5488043, -5.6794167, -9.5190229, -5.7186046, -3.1195469, 3.1330538
1: -13.3072271, -8.7921371, -13.1603060, -8.8123093, -3.4901800, 3.4600358
2: -8.1153669, -4.3342385, -8.0787535, -4.3632770, -3.6815915, 3.6801734
3: -9.8057556, -5.0971947, -9.7819958, -5.1850667, -3.9186020, 3.9104748
4: -11.1214867, -7.0842533, -11.0377111, -7.1098967, -3.5365200, 3.4645367
5: -0.2725630, 3.2816129, -0.2387094, 3.1769609, -3.1460938, 3.1482201
6: 4.4530578, 7.5415411, 4.4925466, 7.4958677, -2.9719520, 2.9325945
7: -18.0521889, -14.3152828, -18.0013618, -14.3410101, -2.9545317, 2.9314394
8: 0.0309204, 4.0946293, 0.1094545, 4.0692034, -3.5873823, 3.5823073
9: -8.8967113, -5.6802301, -8.8810101, -5.7598033, -2.5672736, 2.5412853

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032527, upper bound: 1.2029108
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032527, upper bound: 1.2037859
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.5524578, -5.6769381, -9.5346613, -5.7105474, -3.1329021, 3.1482573
1: -13.3120918, -8.7768087, -13.1846189, -8.7899799, -3.4984112, 3.4938481
2: -8.1187458, -4.3288536, -8.0927105, -4.3504238, -3.6967549, 3.7001157
3: -9.8087893, -5.0934210, -9.7975655, -5.1735635, -3.9323058, 3.9265885
4: -11.1288691, -7.0748668, -11.0600872, -7.0888286, -3.5549393, 3.4931493
5: -0.2802048, 3.2839255, -0.2550375, 3.1854360, -3.1630878, 3.1636243
6: 4.4407496, 7.5424538, 4.4715943, 7.5042624, -2.9894519, 2.9475167
7: -18.0562630, -14.2918262, -18.0248833, -14.3028059, -2.9796181, 2.9785371
8: 0.0262941, 4.0978317, 0.0937520, 4.0806732, -3.5966835, 3.6004205
9: -8.9013004, -5.6698642, -8.8935604, -5.7405472, -2.5904651, 2.5624781

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032526, upper bound: 1.2037187
time: 5.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032526, upper bound: 1.2041120
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5331125, -5.7130709, -9.5594931, -5.7057638, -3.2309275, 3.2460546
1: -13.2007494, -8.8457165, -13.2185516, -8.7746582, -3.4526939, 3.4514327
2: -8.1169844, -4.3479581, -8.1232080, -4.2362223, -3.7421770, 3.6760983
3: -9.7676477, -5.1736517, -9.7809715, -5.1039195, -3.9643850, 3.8688526
4: -11.0554771, -7.1157656, -11.0884037, -7.1123810, -3.5913868, 3.5829468
5: -0.2129452, 3.1904478, -0.2616866, 3.1935134, -3.0951452, 3.1396010
6: 4.4914007, 7.5113392, 4.4475355, 7.5151749, -2.9445524, 2.9476147
7: -18.0344734, -14.3276405, -18.0298786, -14.2983694, -2.9640923, 2.9359012
8: 0.0989506, 4.0742636, 0.0483891, 4.0905895, -3.6350946, 3.6607437
9: -8.8705826, -5.7372036, -8.9347916, -5.7339091, -2.5569110, 2.5491941

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037718, upper bound: 1.2033902
time: 5.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2037862
time: 5.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5367308, -5.7105880, -9.5759029, -5.6977034, -3.2487249, 3.2584620
1: -13.2056026, -8.8303900, -13.2428045, -8.7521267, -3.4627261, 3.4851644
2: -8.1202850, -4.3425713, -8.1372042, -4.2232060, -3.7559400, 3.6966162
3: -9.7706661, -5.1699104, -9.7966595, -5.0923018, -3.9746509, 3.8887796
4: -11.0628662, -7.1063719, -11.1119709, -7.0913124, -3.6141186, 3.6037483
5: -0.2206135, 3.1927545, -0.2780504, 3.2021618, -3.1123362, 3.1552484
6: 4.4791098, 7.5122480, 4.4264622, 7.5235796, -2.9623165, 2.9625287
7: -18.0385437, -14.3041735, -18.0533180, -14.2600193, -2.9875579, 2.9832659
8: 0.0943395, 4.0774326, 0.0326322, 4.1023345, -3.6461735, 3.6787200
9: -8.8751621, -5.7268333, -8.9477205, -5.7146292, -2.5774035, 2.5711603

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2037221
time: 6.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2041124
time: 11.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5582104, -5.6777811, -9.5661411, -5.7042065, -3.2658935, 3.2908297
1: -13.3292103, -8.7859516, -13.2227221, -8.7372522, -3.5284595, 3.5095284
2: -8.1467896, -4.3225851, -8.1311779, -4.2314658, -3.7957726, 3.7254090
3: -9.8088522, -5.0902061, -9.8048916, -5.1002131, -4.0121922, 3.9450779
4: -11.1299887, -7.0758166, -11.0935259, -7.0907674, -3.6465464, 3.6341019
5: -0.2787423, 3.2896318, -0.2945075, 3.1952002, -3.1619468, 3.2013876
6: 4.4469995, 7.5503511, 4.4358497, 7.5171428, -2.9980550, 2.9756064
7: -18.0657043, -14.3078976, -18.0318413, -14.2903757, -2.9877720, 2.9591837
8: 0.0257348, 4.1102757, 0.0430930, 4.1076870, -3.6988940, 3.7078090
9: -8.9032669, -5.6615806, -8.9552088, -5.7272491, -2.5977931, 2.6082778

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041132, upper bound: 1.2033874
time: 5.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2037890
time: 5.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.5619068, -5.6752996, -9.5825624, -5.6961498, -3.2836013, 3.3032341
1: -13.3340664, -8.7706261, -13.2469740, -8.7147284, -3.5384922, 3.5433147
2: -8.1501751, -4.3172007, -8.1451750, -4.2184463, -3.8095369, 3.7459254
3: -9.8118830, -5.0864558, -9.8205719, -5.0885906, -4.0224524, 3.9612198
4: -11.1374178, -7.0664334, -11.1171122, -7.0697036, -3.6692495, 3.6548615
5: -0.2863984, 3.2919426, -0.3108621, 3.2038498, -3.1790991, 3.2170234
6: 4.4346848, 7.5512600, 4.4147592, 7.5255480, -3.0159216, 2.9905243
7: -18.0697670, -14.2844381, -18.0552788, -14.2520256, -3.0112572, 3.0065422
8: 0.0211128, 4.1134834, 0.0273304, 4.1194248, -3.7074385, 3.7257824
9: -8.9078598, -5.6512074, -8.9681311, -5.7079659, -2.6182971, 2.6302433

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 555

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2037191
time: 5.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2041125
time: 5.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.69 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037740, upper bound: 1.2022262
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037718, upper bound: 1.2026264
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2025557
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2029510
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2022262
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2026266
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2025558
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2029510
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2029104
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2037855
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2037182
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2029082, upper bound: 1.2041116
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2032527, upper bound: 1.2029108
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2032527, upper bound: 1.2037859
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2032526, upper bound: 1.2037187
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2032526, upper bound: 1.2041120
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037718, upper bound: 1.2033902
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2037862
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2037221
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2037719, upper bound: 1.2041124
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041132, upper bound: 1.2033874
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2037890
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2037191
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 6, lower bound: -1.2041133, upper bound: 1.2041125

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.4957895, -5.7608871, -9.5446835, -5.7340426, -3.0510254, 3.0919456
1: -13.1013603, -8.9968596, -13.1736202, -8.8564091, -3.2926931, 3.3011794
2: -8.0466194, -4.5005212, -8.0982990, -4.3379221, -3.5621543, 3.4768727
3: -9.6946859, -5.2460442, -9.7584381, -5.1544180, -3.8668408, 3.8106823
4: -10.9398985, -7.2046881, -11.0516472, -7.1743226, -3.2483730, 3.4468250
5: -0.1418009, 3.1617191, -0.2325022, 3.1771486, -3.0117655, 3.1155634
6: 4.5526075, 7.4777136, 4.4713969, 7.4921446, -2.5047064, 2.9063263
7: -18.0047741, -14.3722239, -18.0137043, -14.3157692, -2.9280815, 2.8815942
8: 0.1693759, 4.0088701, 0.0926150, 4.0753031, -3.5043516, 3.4935741
9: -8.7727146, -5.8333464, -8.9215355, -5.8096557, -2.3977594, 2.4948888

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2026072, upper bound: 1.2022255
time: 5.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2026072, upper bound: 1.2022268
time: 5.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5206394, -5.7258658, -9.5512810, -5.7324986, -3.0803165, 3.1359520
1: -13.2291651, -8.9371767, -13.1777916, -8.8190117, -3.3674583, 3.3590651
2: -8.0763493, -4.4754801, -8.1062746, -4.3331718, -3.6154194, 3.5241685
3: -9.7360497, -5.1625462, -9.7823792, -5.1507101, -3.9148254, 3.8982782
4: -11.0125093, -7.1647224, -11.0566635, -7.1526833, -3.3112354, 3.4942150
5: -0.2075830, 3.2603731, -0.2653241, 3.1788261, -3.0784855, 3.1765220
6: 4.5089388, 7.5167637, 4.4597387, 7.4941196, -2.5543270, 2.9471817
7: -18.0359154, -14.3529358, -18.0156593, -14.3077908, -2.9519515, 2.9045229
8: 0.0960450, 4.0448232, 0.0873134, 4.0924397, -3.5733137, 3.5413404
9: -8.8051262, -5.7579880, -8.9419003, -5.8030109, -2.4388032, 2.5569482

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029537, upper bound: 1.2022255
time: 5.24 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029516, upper bound: 1.2022266
time: 6.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5340023, -5.6951308, -9.5552711, -5.7108035, -3.1180673, 3.1445255
1: -13.2693281, -8.8344212, -13.1854553, -8.7434072, -3.4773474, 3.3646295
2: -8.1185055, -4.3629465, -8.1160517, -4.2476478, -3.7388315, 3.5517845
3: -9.7569485, -5.1526518, -9.7959061, -5.1468115, -3.9288731, 3.9195786
4: -11.0413628, -7.1487184, -11.0768414, -7.1469383, -3.3412437, 3.5299635
5: -0.2367301, 3.2691214, -0.2845085, 3.1818337, -3.1075044, 3.2037470
6: 4.4940901, 7.5242715, 4.4521132, 7.4993196, -2.5812001, 2.9597039
7: -18.0490303, -14.3336754, -18.0222340, -14.2985439, -2.9643593, 2.9466343
8: 0.0741022, 4.0624819, 0.0725411, 4.0986786, -3.5866270, 3.5705857
9: -8.8209324, -5.7287445, -8.9473515, -5.7828546, -2.4776430, 2.5702422

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029516, upper bound: 1.2026259
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029515, upper bound: 1.2026270
time: 5.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5242052, -5.7236214, -9.5668383, -5.7243586, -3.0940752, 3.1507730
1: -13.2334175, -8.9215393, -13.2020874, -8.7964983, -3.3723741, 3.3984556
2: -8.0796280, -4.4704123, -8.1202126, -4.3202591, -3.6294270, 3.5438433
3: -9.7390327, -5.1588092, -9.7980013, -5.1390729, -3.9249773, 3.9144039
4: -11.0189075, -7.1552687, -11.0792942, -7.1312733, -3.3316231, 3.5233421
5: -0.2153611, 3.2623720, -0.2817760, 3.1872985, -3.0958519, 3.1922386
6: 4.4974337, 7.5177093, 4.4386735, 7.5025086, -2.5747051, 2.9621382
7: -18.0401268, -14.3295918, -18.0389500, -14.2693844, -2.9749174, 2.9516926
8: 0.0914874, 4.0480700, 0.0716008, 4.1039042, -3.5825815, 3.5594401
9: -8.8097734, -5.7474513, -8.9543419, -5.7834039, -2.4650373, 2.5781624

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029515, upper bound: 1.2025552
time: 5.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029518, upper bound: 1.2025581
time: 5.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.5376501, -5.6929941, -9.5709181, -5.7027292, -3.1314559, 3.1579623
1: -13.2735605, -8.8191357, -13.2097588, -8.7210636, -3.4822221, 3.4001153
2: -8.1218472, -4.3579588, -8.1300850, -4.2347593, -3.7526979, 3.5714536
3: -9.7599497, -5.1489301, -9.8115120, -5.1351757, -3.9406443, 3.9357119
4: -11.0473785, -7.1393247, -11.0992641, -7.1258626, -3.3614278, 3.5575171
5: -0.2443676, 3.2710779, -0.3008695, 3.1902981, -3.1245875, 3.2193370
6: 4.4825573, 7.5251794, 4.4310865, 7.5077043, -2.6015515, 2.9745581
7: -18.0531063, -14.3104038, -18.0456696, -14.2603531, -2.9873371, 2.9933000
8: 0.0695214, 4.0656629, 0.0567745, 4.1101894, -3.5957441, 3.5887051
9: -8.8254728, -5.7185640, -8.9598446, -5.7635951, -2.5033741, 2.5912201

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029518, upper bound: 1.2029497
time: 5.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029518, upper bound: 1.2029511
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.5682774, -5.7025881, -9.5124226, -5.7201567, -3.1342235, 3.1017776
1: -13.2285986, -8.7794914, -13.1560402, -8.8496752, -3.4520679, 3.4233093
2: -8.1302280, -4.2427959, -8.0707684, -4.3680010, -3.6657171, 3.6952591
3: -9.7844753, -5.1227207, -9.7580681, -5.1887374, -3.8884726, 3.8902206
4: -11.0974569, -7.1080818, -11.0326548, -7.1315556, -3.5154686, 3.4352956
5: -0.2585912, 3.1977684, -0.2059362, 3.1752872, -3.1300001, 3.0845008
6: 4.4504757, 7.5208998, 4.5041285, 7.4938841, -2.9481182, 2.9042459
7: -18.0457153, -14.2867985, -17.9993935, -14.3489714, -2.9385347, 2.9485426
8: 0.0414915, 4.0901799, 0.1147404, 4.0521059, -3.5518131, 3.5545125
9: -8.9365044, -5.7316551, -8.8606577, -5.7664642, -2.5596533, 2.4883654

Time for backsubstitution: 14.61 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.956814765930176
rel_dist={6: [-1.2041204236983045, 1.204119556210955]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 2420.12 seconds
