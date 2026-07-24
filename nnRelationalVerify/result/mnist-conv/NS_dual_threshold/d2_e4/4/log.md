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
execution time: IAR + RelationalAnalysis = 22.55 + 38.83 = 61.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8520508, upper bound: 0.8520506

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6195
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8465318
time: 15.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8520469
time: 40.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 56.95 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 56.95
Output dim: 8, lower bound: -0.8520470, upper bound: 0.8465318
NS_A2, status: Status.UNKNOWN, split count: 1, time: 56.95
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

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8485432, upper bound: 0.8457823
time: 6.29 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520425, upper bound: 0.8465291
time: 10.49 seconds

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

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8485431, upper bound: 0.8513047
time: 6.40 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520425, upper bound: 0.8520426
time: 7.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 35.95 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 35.95
Output dim: 8, lower bound: -0.8485432, upper bound: 0.8457823
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 35.95
Output dim: 8, lower bound: -0.8520425, upper bound: 0.8465291
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 35.95
Output dim: 8, lower bound: -0.8485431, upper bound: 0.8513047
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 35.95
Output dim: 8, lower bound: -0.8520425, upper bound: 0.8520426

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -5.3856859, -2.5973811, -5.3956528, -2.5874648, -2.3978558, 2.3952980
1: -14.4865589, -11.6210117, -14.5001850, -11.6016140, -2.4411592, 2.4322858
2: -8.4281178, -5.6352472, -8.4576893, -5.5788078, -2.1529837, 2.1203947
3: -6.8000088, -4.3024840, -6.8147016, -4.2780285, -2.2894192, 2.2792511
4: -11.1486759, -8.1637335, -11.1841183, -8.1497517, -2.7387466, 2.7527447
5: -5.3180771, -2.9306264, -5.3418660, -2.9167252, -1.8801851, 1.8824124
6: -12.9931545, -10.1612186, -13.0143404, -10.1459808, -1.9452839, 1.9523344
7: -9.3980465, -6.6571946, -9.4110355, -6.6510935, -2.5420265, 2.5410395
8: 8.5960808, 10.6240673, 8.5702457, 10.6357594, -1.4254236, 1.4502964
9: -6.3079944, -3.9264417, -6.3233981, -3.9212666, -1.7693658, 1.7727447

Time for backsubstitution: 22.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4555

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514786, upper bound: 0.8465273
time: 6.31 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520416, upper bound: 0.8465268
time: 7.39 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -5.3570595, -2.6069202, -5.3776150, -2.5901248, -2.3617420, 2.3969021
1: -14.4687252, -11.6170940, -14.4880466, -11.6043749, -2.4129925, 2.4372387
2: -8.4896832, -5.6025448, -8.4948540, -5.5903997, -2.1128111, 2.1580868
3: -6.8173461, -4.2829456, -6.8245978, -4.2793961, -2.2842045, 2.3092060
4: -11.1663113, -8.1518183, -11.1818867, -8.1433973, -2.7635660, 2.7336235
5: -5.3308439, -2.9125721, -5.3389044, -2.9070277, -1.8882475, 1.8693867
6: -12.9863729, -10.1454763, -12.9997082, -10.1342497, -1.9423265, 1.9274063
7: -9.3990269, -6.6646695, -9.4093552, -6.6574602, -2.5264864, 2.5530224
8: 8.5899048, 10.6367130, 8.5796843, 10.6464224, -1.4431548, 1.4204321
9: -6.3071375, -3.9318953, -6.3168435, -3.9248781, -1.7586675, 1.7631717

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8485423, upper bound: 0.8507481
time: 12.58 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8485423, upper bound: 0.8513054
time: 8.12 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -5.4006190, -2.5856276, -5.4006367, -2.5856211, -2.4069443, 2.4393129
1: -14.5097380, -11.6001701, -14.5097599, -11.6001663, -2.4584484, 2.4795260
2: -8.4967136, -5.5774937, -8.4967422, -5.5774856, -2.1360350, 2.1832576
3: -6.8286800, -4.2764816, -6.8286886, -4.2764773, -2.2981777, 2.3199482
4: -11.1889687, -8.1335154, -11.1889763, -8.1334991, -2.7987709, 2.7522764
5: -5.3435702, -2.9016926, -5.3435726, -2.9016798, -1.9135509, 1.8875475
6: -13.0153370, -10.1305008, -13.0153484, -10.1304855, -1.9730883, 1.9585023
7: -9.4142294, -6.6494617, -9.4142342, -6.6494551, -2.5504656, 2.5715408
8: 8.5683765, 10.6499014, 8.5683670, 10.6499062, -1.4572399, 1.4443817
9: -6.3254099, -3.9187260, -6.3254161, -3.9187231, -1.7906270, 1.7832437

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 4555

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520417, upper bound: 0.8514790
time: 7.30 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520417, upper bound: 0.8520418
time: 8.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.54 seconds
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 38.54
Output dim: 8, lower bound: -0.8514786, upper bound: 0.8465273
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 38.54
Output dim: 8, lower bound: -0.8520416, upper bound: 0.8465268
NS_A2_A1_A1, status: Status.VERIFIED, split count: 3, time: 38.54
Output dim: 8, lower bound: -0.8485423, upper bound: 0.8507481
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 38.54
Output dim: 8, lower bound: -0.8485423, upper bound: 0.8513054
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 38.54
Output dim: 8, lower bound: -0.8520417, upper bound: 0.8514790
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 38.54
Output dim: 8, lower bound: -0.8520417, upper bound: 0.8520418

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5.3828535, -2.5998178, -5.3889413, -2.6028037, -2.3772697, 2.3843460
1: -14.4852161, -11.6225147, -14.4928513, -11.6066837, -2.4285984, 2.4206905
2: -8.4272394, -5.6381636, -8.4526711, -5.5867953, -2.1436014, 2.1120129
3: -6.7983737, -4.3151808, -6.7974086, -4.3016043, -2.2630854, 2.2474546
4: -11.1445122, -8.1647549, -11.1732731, -8.1559610, -2.7265291, 2.7399330
5: -5.3170033, -2.9334183, -5.3353786, -2.9227142, -1.8723955, 1.8723927
6: -12.9922934, -10.1805515, -12.9994516, -10.1820087, -1.9091778, 1.9138274
7: -9.3853569, -6.6584377, -9.3849792, -6.6643438, -2.5108738, 2.5129499
8: 8.5975142, 10.6206083, 8.5779839, 10.6283970, -1.4159579, 1.4370341
9: -6.3060446, -3.9274409, -6.3129072, -3.9238634, -1.7636652, 1.7600965

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4555

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 581

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514777, upper bound: 0.8456348
time: 9.27 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514777, upper bound: 0.8465276
time: 9.98 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5.3856850, -2.5973816, -5.3956513, -2.5874691, -2.4030905, 2.3938932
1: -14.4865570, -11.6210117, -14.5001888, -11.6016150, -2.4400311, 2.4310713
2: -8.4281149, -5.6352496, -8.4576883, -5.5788102, -2.1529789, 2.1204290
3: -6.8000102, -4.3024888, -6.8146996, -4.2780342, -2.2655392, 2.2792492
4: -11.1486750, -8.1637335, -11.1841164, -8.1497498, -2.7387457, 2.7511272
5: -5.3180757, -2.9306278, -5.3418632, -2.9167290, -1.8783598, 1.8824129
6: -12.9931526, -10.1612244, -13.0143385, -10.1459866, -1.9132357, 1.9449263
7: -9.3980436, -6.6571932, -9.4110317, -6.6510954, -2.5420237, 2.5203695
8: 8.5960827, 10.6240654, 8.5702477, 10.6357584, -1.4203572, 1.4502940
9: -6.3079929, -3.9264398, -6.3233995, -3.9212687, -1.7691598, 1.7725701

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 536

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 581

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8456365
time: 8.96 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8465276
time: 7.44 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -5.3570600, -2.6069198, -5.3776116, -2.5901260, -2.3603516, 2.4021091
1: -14.4687262, -11.6170940, -14.4880476, -11.6043758, -2.4117289, 2.4330611
2: -8.4896822, -5.6025438, -8.4948549, -5.5903997, -2.1128473, 2.1570625
3: -6.8173461, -4.2829514, -6.8245959, -4.2793984, -2.2842007, 2.2842774
4: -11.1663074, -8.1518183, -11.1818848, -8.1433983, -2.7619219, 2.7336226
5: -5.3308430, -2.9125757, -5.3389063, -2.9070277, -1.8858919, 1.8675232
6: -12.9863739, -10.1454849, -12.9997110, -10.1342535, -1.9292998, 1.8936377
7: -9.3990173, -6.6646705, -9.4093533, -6.6574607, -2.5057468, 2.5516138
8: 8.5899076, 10.6367111, 8.5796824, 10.6464195, -1.4400666, 1.4153242
9: -6.3071365, -3.9318964, -6.3168449, -3.9248817, -1.7584929, 1.7629662

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 536

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 581

## Relational analysis of NS_A2_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8485414, upper bound: 0.8504106
time: 9.05 seconds

## Relational analysis of NS_A2_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8485414, upper bound: 0.8513029
time: 9.93 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -5.3939023, -2.6009650, -5.3978033, -2.5881205, -2.3959427, 2.4187422
1: -14.5024166, -11.6052580, -14.5083971, -11.6017008, -2.4467831, 2.4670315
2: -8.4917021, -5.5854807, -8.4958477, -5.5804415, -2.1276236, 2.1738863
3: -6.8113861, -4.3000507, -6.8270431, -4.2891707, -2.2663841, 2.2936153
4: -11.1781092, -8.1397142, -11.1846085, -8.1345339, -2.7859278, 2.7397661
5: -5.3370934, -2.9076860, -5.3425040, -2.9044819, -1.9035153, 1.8797584
6: -13.0004482, -10.1665363, -13.0144854, -10.1498575, -1.9306703, 1.9223890
7: -9.3881588, -6.6627007, -9.4013672, -6.6506982, -2.5223427, 2.5402060
8: 8.5761070, 10.6425371, 8.5697975, 10.6464119, -1.4434128, 1.4349232
9: -6.3149366, -3.9213238, -6.3233619, -3.9197218, -1.7780290, 1.7774467

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 4555

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 581

## Relational analysis of NS_A2_A2_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8505892
time: 12.74 seconds

## Relational analysis of NS_A2_A2_A1_A2

### Relational analysis result of NS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8514776
time: 9.46 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -5.4006157, -2.5856280, -5.4006338, -2.5856214, -2.4055490, 2.4445219
1: -14.5097399, -11.6001701, -14.5097570, -11.6001663, -2.4572315, 2.4752827
2: -8.4967117, -5.5774946, -8.4967422, -5.5774837, -2.1360683, 2.1822329
3: -6.8286777, -4.2764845, -6.8286886, -4.2764812, -2.2981749, 2.2947168
4: -11.1889668, -8.1335163, -11.1889772, -8.1335020, -2.7971535, 2.7522745
5: -5.3435717, -2.9016967, -5.3435736, -2.9016824, -1.9111958, 1.8857193
6: -13.0153370, -10.1305084, -13.0153475, -10.1304893, -1.9600601, 1.9252844
7: -9.4142227, -6.6494617, -9.4142313, -6.6494551, -2.5297804, 2.5688686
8: 8.5683765, 10.6498985, 8.5683689, 10.6499062, -1.4541512, 1.4393096
9: -6.3254075, -3.9187281, -6.3254132, -3.9187241, -1.7904520, 1.7830367

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 581

## Relational analysis of NS_A2_A2_A2_A1

### Relational analysis result of NS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8511523
time: 19.74 seconds

## Relational analysis of NS_A2_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8520408
time: 9.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 51.48 seconds
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8514777, upper bound: 0.8456348
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8514777, upper bound: 0.8465276
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8456365
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8465276
NS_A2_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8485414, upper bound: 0.8504106
NS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8485414, upper bound: 0.8513029
NS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8505892
NS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8514776
NS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8511523
NS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 51.48
Output dim: 8, lower bound: -0.8520407, upper bound: 0.8520408

## BFS NS instance: NS_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.3779302, -2.6059163, -5.3881826, -2.6059110, -2.3691568, 2.3776026
1: -14.4765797, -11.6267700, -14.4881821, -11.6068907, -2.4195042, 2.4109259
2: -8.4260254, -5.6428514, -8.4522266, -5.5883274, -2.1388874, 2.1061096
3: -6.7934065, -4.3194880, -6.7952266, -4.3024249, -2.2567778, 2.2407370
4: -11.1388712, -8.1669712, -11.1712198, -8.1561022, -2.7204523, 2.7349510
5: -5.3126392, -2.9373741, -5.3333049, -2.9233265, -1.8673677, 1.8662767
6: -12.9812012, -10.1880856, -12.9929724, -10.1823750, -1.8975964, 1.8995752
7: -9.3755360, -6.6648755, -9.3835173, -6.6680980, -2.4971972, 2.5050907
8: 8.6016150, 10.6162567, 8.5783310, 10.6260757, -1.4089088, 1.4322467
9: -6.3017497, -3.9298625, -6.3111072, -3.9244990, -1.7587638, 1.7555313

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4555

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of NS_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8459630, upper bound: 0.8456366
time: 5.93 seconds

## Relational analysis of NS_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8459630, upper bound: 0.8456352
time: 5.90 seconds

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.3828502, -2.5998204, -5.3889399, -2.6028056, -2.3772678, 2.3820858
1: -14.4852161, -11.6225128, -14.4928513, -11.6066837, -2.4223337, 2.4206390
2: -8.4272375, -5.6381640, -8.4526711, -5.5867968, -2.1437073, 2.1100378
3: -6.7983704, -4.3151798, -6.7974095, -4.3016057, -2.2629166, 2.2454977
4: -11.1445103, -8.1647539, -11.1732712, -8.1559601, -2.7259665, 2.7398310
5: -5.3170023, -2.9334190, -5.3353786, -2.9227157, -1.8712893, 1.8723936
6: -12.9922876, -10.1805506, -12.9994507, -10.1820087, -1.8960896, 1.9090924
7: -9.3853579, -6.6584425, -9.3849783, -6.6643467, -2.5108719, 2.5058012
8: 8.5975161, 10.6206074, 8.5779858, 10.6283960, -1.4155307, 1.4334798
9: -6.3060427, -3.9274428, -6.3129077, -3.9238622, -1.7628646, 1.7613730

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 4555

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of NS_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8459630, upper bound: 0.8465276
time: 6.85 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8459630, upper bound: 0.8465276
time: 7.86 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3807654, -2.6034818, -5.3948994, -2.5905771, -2.3949776, 2.3871479
1: -14.4779167, -11.6252708, -14.4955130, -11.6018229, -2.4309492, 2.4212990
2: -8.4268970, -5.6399307, -8.4572401, -5.5803328, -2.1482744, 2.1145296
3: -6.7950430, -4.3067913, -6.8125176, -4.2788525, -2.2592363, 2.2725353
4: -11.1430244, -8.1659460, -11.1820660, -8.1498947, -2.7326756, 2.7461338
5: -5.3137126, -2.9345853, -5.3397870, -2.9173388, -1.8733339, 1.8763022
6: -12.9820576, -10.1687584, -13.0078573, -10.1463585, -1.9016533, 1.9289432
7: -9.3882284, -6.6636305, -9.4095526, -6.6548529, -2.5283499, 2.5125008
8: 8.6001778, 10.6197138, 8.5705910, 10.6334410, -1.4133015, 1.4455099
9: -6.3036981, -3.9288607, -6.3215928, -3.9219019, -1.7642603, 1.7680049

Time for backsubstitution: 22.23 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 61.37 + 559.61 = 620.98 seconds
