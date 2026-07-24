## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.8511987492


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4265327, 2.4265337)
1: (-14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4702597, 2.4702601)
2: (-8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2232647, 2.2232647)
3: (-6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3200216, 2.3200216)
4: (-11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.8002758, 2.8002748)
5: (-5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9139080, 1.9139075)
6: (-13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9840040, 1.9840040)
7: (-9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5707560, 2.5707560)
8: (8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4787288, 1.4787283)
9: (-6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7844839, 1.7844839)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.54 + 39.90 = 63.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8520508, upper bound: 0.8520506

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8465318
time: 16.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8520469
time: 42.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 58.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 58.94
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8465318
NS_A2, status: Status.UNKNOWN, split count: 1, time: 58.94
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8520469

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.3857098, -2.5973783, -5.3956652, -2.5874641, -2.3988495, 2.3953114
1: -14.4865847, -11.6210079, -14.5001984, -11.6016140, -2.4415150, 2.4294877
2: -8.4281178, -5.6352305, -8.4576893, -5.5788021, -2.1529942, 2.1259356
3: -6.8000154, -4.3024807, -6.8147025, -4.2780275, -2.2889166, 2.2793179
4: -11.1486845, -8.1637278, -11.1841221, -8.1497498, -2.7387590, 2.7638121
5: -5.3180823, -2.9306211, -5.3418660, -2.9167254, -1.8732204, 1.8832426
6: -12.9931755, -10.1612158, -13.0143499, -10.1459770, -1.9457445, 1.9523473
7: -9.3980522, -6.6571822, -9.4110374, -6.6510868, -2.5420370, 2.5431938
8: 8.5960665, 10.6240711, 8.5702410, 10.6357622, -1.4348941, 1.4503059
9: -6.3080053, -3.9264374, -6.3234015, -3.9212656, -1.7632174, 1.7734818

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8465311
time: 8.99 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8465328
time: 6.41 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.4006438, -2.5856233, -5.4006433, -2.5856242, -2.4079390, 2.4393291
1: -14.5097656, -11.6001644, -14.5097666, -11.6001625, -2.4588060, 2.4838309
2: -8.4967165, -5.5774775, -8.4967403, -5.5774775, -2.1360445, 2.1891685
3: -6.8286810, -4.2764773, -6.8286905, -4.2764769, -2.2976770, 2.3200130
4: -11.1889782, -8.1335106, -11.1889820, -8.1334972, -2.7987833, 2.7633438
5: -5.3435731, -2.9016888, -5.3435740, -2.9016807, -1.9100275, 1.8883791
6: -13.0153580, -10.1304989, -13.0153580, -10.1304836, -1.9760427, 1.9585142
7: -9.4142351, -6.6494532, -9.4142361, -6.6494517, -2.5504799, 2.5736952
8: 8.5683632, 10.6499062, 8.5683603, 10.6499081, -1.4680877, 1.4443922
9: -6.3254175, -3.9187238, -6.3254175, -3.9187212, -1.7844791, 1.7839794

Time for backsubstitution: 22.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8520467
time: 11.13 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8520466
time: 17.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 51.95 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 51.95
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8465311
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 51.95
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8465328
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 51.95
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8520467
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 51.95
Output dim: 8, lower bound: -0.8465314, upper bound: 0.8520466

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.4006438, -2.5856233, -5.3857098, -2.5973783, -2.3881674, 2.3863325
1: -14.5097656, -11.6001644, -14.4865847, -11.6210079, -2.4306364, 2.4314389
2: -8.4967165, -5.5774775, -8.4281178, -5.6352305, -2.1359110, 2.1203675
3: -6.8286810, -4.2764773, -6.8000154, -4.3024807, -2.2940416, 2.2907887
4: -11.1889782, -8.1335106, -11.1486845, -8.1637278, -2.7670259, 2.7584248
5: -5.3435731, -2.9016888, -5.3180823, -2.9306211, -1.8810949, 1.8876286
6: -13.0153580, -10.1304989, -12.9931755, -10.1612158, -1.9454689, 1.9567318
7: -9.4142351, -6.6494532, -9.3980522, -6.6571822, -2.5354204, 2.5281286
8: 8.5683632, 10.6499062, 8.5960665, 10.6240711, -1.4417872, 1.4423454
9: -6.3254175, -3.9187238, -6.3080053, -3.9264374, -1.7759004, 1.7665949

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8430210, upper bound: 0.8513039
time: 9.40 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8465269, upper bound: 0.8520433
time: 6.87 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.4006438, -2.5856233, -5.4006438, -2.5856233, -2.4393196, 2.4393196
1: -14.5097656, -11.6001644, -14.5097656, -11.6001644, -2.4839649, 2.4838281
2: -8.4967165, -5.5774775, -8.4967165, -5.5774775, -2.1360445, 2.1360445
3: -6.8286810, -4.2764773, -6.8286810, -4.2764773, -2.2976742, 2.2976742
4: -11.1889782, -8.1335106, -11.1889782, -8.1335106, -2.7633410, 2.7633410
5: -5.3435731, -2.9016888, -5.3435731, -2.9016888, -1.8883772, 1.8883772
6: -13.0153580, -10.1304989, -13.0153580, -10.1304989, -1.9585133, 1.9585137
7: -9.4142351, -6.6494532, -9.4142351, -6.6494532, -2.5736847, 2.5736847
8: 8.5683632, 10.6499062, 8.5683632, 10.6499062, -1.4443903, 1.4443903
9: -6.3254175, -3.9187238, -6.3254175, -3.9187238, -1.7839789, 1.7839789

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8430210, upper bound: 0.8513050
time: 8.22 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8465269, upper bound: 0.8520444
time: 7.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.26 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 38.26
Output dim: 8, lower bound: -0.8430210, upper bound: 0.8513039
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 38.26
Output dim: 8, lower bound: -0.8465269, upper bound: 0.8520433
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.26
Output dim: 8, lower bound: -0.8430210, upper bound: 0.8513050
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.26
Output dim: 8, lower bound: -0.8465269, upper bound: 0.8520444

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.3570595, -2.6069202, -5.3626490, -2.6019011, -2.3419733, 2.3438978
1: -14.4687252, -11.6170940, -14.4648228, -11.6251993, -2.3848438, 2.3906884
2: -8.4896832, -5.6025448, -8.4262238, -5.6481371, -2.1100101, 2.0892839
3: -6.8173461, -4.2829456, -6.7959743, -4.3053966, -2.2805834, 2.2800121
4: -11.1663113, -8.1518183, -11.1416473, -8.1736259, -2.7318001, 2.7288980
5: -5.3308439, -2.9125721, -5.3134050, -2.9359751, -1.8593035, 1.8685427
6: -12.9863729, -10.1454763, -12.9775391, -10.1649275, -1.9117708, 1.9197936
7: -9.3990269, -6.6646695, -9.3932600, -6.6651936, -2.5114260, 2.5075102
8: 8.5899048, 10.6367130, 8.6073847, 10.6205740, -1.4169254, 1.4131789
9: -6.3071375, -3.9318953, -6.2994676, -3.9326015, -1.7500515, 1.7458873

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 581

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8421276, upper bound: 0.8513031
time: 9.42 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8430207, upper bound: 0.8513030
time: 18.39 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.4006190, -2.5856276, -5.3856993, -2.5973778, -2.3871727, 2.3863163
1: -14.5097380, -11.6001701, -14.4865723, -11.6210108, -2.4302807, 2.4342341
2: -8.4967136, -5.5774937, -8.4281187, -5.6352396, -2.1336308, 2.1144552
3: -6.8286800, -4.2764816, -6.8000126, -4.3024836, -2.2945442, 2.2907219
4: -11.1889687, -8.1335154, -11.1486816, -8.1637306, -2.7670135, 2.7473574
5: -5.3435702, -2.9016926, -5.3180795, -2.9306254, -1.8846164, 1.8859224
6: -13.0153370, -10.1305008, -12.9931660, -10.1612148, -1.9425182, 1.9514675
7: -9.4142294, -6.6494617, -9.3980494, -6.6571870, -2.5354099, 2.5259714
8: 8.5683765, 10.6499014, 8.5960751, 10.6240692, -1.4309378, 1.4369240
9: -6.3254099, -3.9187260, -6.3079991, -3.9264374, -1.7820468, 1.7658567

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8457819, upper bound: 0.8485418
time: 5.82 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8457819, upper bound: 0.8485420
time: 6.70 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3570595, -2.6069202, -5.3776107, -2.5901275, -2.3931236, 2.3968925
1: -14.4687252, -11.6170940, -14.4880457, -11.6043739, -2.4381495, 2.4372349
2: -8.4896832, -5.6025448, -8.4948254, -5.5904021, -2.1128092, 2.1048937
3: -6.8173461, -4.2829456, -6.8245864, -4.2793980, -2.2842035, 2.2868633
4: -11.1663113, -8.1518183, -11.1818848, -8.1434107, -2.7281237, 2.7336206
5: -5.3308439, -2.9125721, -5.3389039, -2.9070368, -1.8666086, 1.8693852
6: -12.9863729, -10.1454763, -12.9997063, -10.1342621, -1.9241638, 1.9264526
7: -9.3990269, -6.6646695, -9.4093561, -6.6574612, -2.5498009, 2.5530109
8: 8.5899048, 10.6367130, 8.5796843, 10.6464186, -1.4195781, 1.4204307
9: -6.3071375, -3.9318953, -6.3168406, -3.9248829, -1.7581668, 1.7631702

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 581

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8421270, upper bound: 0.8513041
time: 22.80 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8430200, upper bound: 0.8513041
time: 6.48 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.4006190, -2.5856276, -5.4006324, -2.5856261, -2.4383268, 2.4393044
1: -14.5097380, -11.6001701, -14.5097542, -11.6001682, -2.4836073, 2.4795227
2: -8.4967136, -5.5774937, -8.4967127, -5.5774870, -2.1360340, 2.1305008
3: -6.8286800, -4.2764816, -6.8286800, -4.2764778, -2.2981768, 2.2976074
4: -11.1889687, -8.1335154, -11.1889763, -8.1335115, -2.7633305, 2.7522736
5: -5.3435702, -2.9016926, -5.3435736, -2.9016917, -1.8953409, 1.8875465
6: -13.0153370, -10.1305008, -13.0153513, -10.1304970, -1.9580526, 1.9581060
7: -9.4142294, -6.6494617, -9.4142342, -6.6494575, -2.5736723, 2.5715294
8: 8.5683765, 10.6499014, 8.5683670, 10.6499043, -1.4349198, 1.4443808
9: -6.3254099, -3.9187260, -6.3254151, -3.9187236, -1.7901258, 1.7832413

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8457813, upper bound: 0.8485423
time: 5.46 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8457813, upper bound: 0.8485428
time: 12.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 39.48 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8421276, upper bound: 0.8513031
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8430207, upper bound: 0.8513030
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8457819, upper bound: 0.8485418
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8457819, upper bound: 0.8485420
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8421270, upper bound: 0.8513041
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8430200, upper bound: 0.8513041
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8457813, upper bound: 0.8485423
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 39.48
Output dim: 8, lower bound: -0.8457813, upper bound: 0.8485428

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.3562946, -2.6100335, -5.3577185, -2.6079931, -2.3352518, 2.3357716
1: -14.4641266, -11.6173058, -14.4562597, -11.6294603, -2.3751125, 2.3816805
2: -8.4892349, -5.6040740, -8.4250126, -5.6528544, -2.1041536, 2.0845942
3: -6.8151875, -4.2837515, -6.7910218, -4.3096972, -2.2738934, 2.2737226
4: -11.1642590, -8.1519585, -11.1360331, -8.1758423, -2.7267876, 2.7228174
5: -5.3287601, -2.9131794, -5.3090477, -2.9399400, -1.8526859, 1.8634849
6: -12.9799299, -10.1458321, -12.9664869, -10.1724520, -1.8958035, 1.9082236
7: -9.3975248, -6.6684237, -9.3834114, -6.6716290, -2.5035276, 2.4938078
8: 8.5902433, 10.6344070, 8.6114845, 10.6162405, -1.4121478, 1.4061649
9: -6.3053422, -3.9325366, -6.2952046, -3.9350300, -1.7454658, 1.7410235

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8421268, upper bound: 0.8452241
time: 16.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8421268, upper bound: 0.8513022
time: 9.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.3570604, -2.6069207, -5.3626490, -2.6019063, -2.3397150, 2.3438969
1: -14.4687243, -11.6170921, -14.4648199, -11.6252003, -2.3847914, 2.3845119
2: -8.4896812, -5.6025438, -8.4262238, -5.6481400, -2.1068287, 2.0874243
3: -6.8173447, -4.2829456, -6.7959733, -4.3053994, -2.2786093, 2.2777028
4: -11.1663074, -8.1518192, -11.1416473, -8.1736288, -2.7306299, 2.7283354
5: -5.3308430, -2.9125750, -5.3134041, -2.9359736, -1.8568726, 1.8661184
6: -12.9863701, -10.1454773, -12.9775333, -10.1649265, -1.9053073, 1.9063082
7: -9.3990250, -6.6646705, -9.3932590, -6.6651955, -2.5042810, 2.5075083
8: 8.5899076, 10.6367121, 8.6073875, 10.6205730, -1.4123082, 1.4100368
9: -6.3071356, -3.9318950, -6.2994680, -3.9326015, -1.7512040, 1.7450156

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8430200, upper bound: 0.8507465
time: 12.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8430199, upper bound: 0.8513021
time: 8.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.3562946, -2.6100335, -5.3727188, -2.5966587, -2.3859663, 2.3888025
1: -14.4641266, -11.6173058, -14.4794760, -11.6091099, -2.4278851, 2.4282079
2: -8.4892349, -5.6040740, -8.4936094, -5.5954447, -2.1066313, 2.1001897
3: -6.8151875, -4.2837515, -6.8196211, -4.2837391, -2.2774611, 2.2806101
4: -11.1642590, -8.1519585, -11.1745834, -8.1456223, -2.7231226, 2.7256432
5: -5.3287601, -2.9131794, -5.3345456, -2.9110532, -1.8604126, 1.8643584
6: -12.9799299, -10.1458321, -12.9886513, -10.1419888, -1.9099064, 1.9148850
7: -9.3975248, -6.6684237, -9.3982668, -6.6638989, -2.5418997, 2.5381136
8: 8.5902433, 10.6344070, 8.5837822, 10.6418066, -1.4145603, 1.4134383
9: -6.3053422, -3.9325366, -6.3117871, -3.9272881, -1.7535911, 1.7576413

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8421261, upper bound: 0.8507471
time: 7.03 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8421262, upper bound: 0.8513028
time: 6.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.3570604, -2.6069207, -5.3776112, -2.5901270, -2.3908653, 2.3968897
1: -14.4687243, -11.6170921, -14.4880428, -11.6043749, -2.4381046, 2.4305019
2: -8.4896812, -5.6025438, -8.4948244, -5.5904031, -2.1108618, 2.1049337
3: -6.8173447, -4.2829456, -6.8245869, -4.2793980, -2.2822256, 2.2866793
4: -11.1663074, -8.1518192, -11.1818848, -8.1434135, -2.7279654, 2.7330580
5: -5.3308430, -2.9125750, -5.3389010, -2.9070368, -1.8666105, 1.8682818
6: -12.9863701, -10.1454773, -12.9997063, -10.1342611, -1.9241581, 1.9129686
7: -9.3990250, -6.6646705, -9.4093552, -6.6574650, -2.5426540, 2.5530100
8: 8.5899076, 10.6367121, 8.5796833, 10.6464167, -1.4160714, 1.4187217
9: -6.3071356, -3.9318950, -6.3168397, -3.9248824, -1.7593207, 1.7622900

Time for backsubstitution: 21.26 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 63.44 + 548.19 = 611.63 seconds
