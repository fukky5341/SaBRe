## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 46.318565822800004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495)
1: (-17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504)
2: (-17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372)
3: (-22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624)
4: (-20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585)

## BASE Result
execution time: IAR + LP analysis = 2.49 + 1.87 = 4.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -46.4113886, upper bound: 46.4113886


# Binary Search by BASE starts (time budget: 1195.64 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=55.00176239013672
rel_dist={3: [-46.41123857032521, 46.41123857032903]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=55.00176239013672
rel_dist={3: [-46.410039219938405, 46.4100392199384]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=55.00176239013672
rel_dist={3: [-46.40791696364876, 46.40791696367562]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=55.00176239013672
rel_dist={3: [-46.40672265181321, 46.40672265182802]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=55.00176239013672
rel_dist={3: [-46.40611311757378, 46.406113117581526]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=55.00176239013672
rel_dist={3: [-46.40580581063882, 46.40580581064276]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=55.00176239013672
rel_dist={3: [-46.405651916648736, 46.40565189290891]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=55.00176239013672
rel_dist={3: [-46.405574448472976, 46.4055744584641]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=55.00176239013672
rel_dist={3: [-46.40553548314551, 46.40553551008645]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=55.00176239013672
rel_dist={3: [-46.40551600158139, 46.405516036735335]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=55.00176239013672
rel_dist={3: [-46.40550626293548, 46.4055063016896]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=55.00176239013672
rel_dist={3: [-46.405501397303595, 46.405501397303524]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=55.00176239013672
rel_dist={3: [-46.40549902064443, 46.40549902136124]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=55.00176239013672
rel_dist={3: [-46.40549778989501, 46.4054978543421]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=55.00176239013672
rel_dist={3: [-46.40549734791982, 46.405497434778894]}

## Binary Search Result
Binary search time: 76.85 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1118.79 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
time: 0.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4113240, upper bound: 46.4113238
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -46.4113240, upper bound: 46.4113238

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -15.6748066, 27.2767467, -40.5789375, 39.4772720
1: -15.2836475, 23.8005772, -17.9646091, 27.3548412, -42.6384850, 41.7651787
2: -15.1096706, 23.1136723, -17.7045784, 26.5651569, -41.6748276, 40.8182526
3: -19.4732094, 28.0227852, -22.8100491, 32.1917152, -51.6649170, 50.8328323
4: -17.4002285, 26.2846336, -20.1704769, 30.4162006, -47.8164215, 46.4551048

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4032405
time: 0.98 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -15.6724682, 27.2734356, -49.4864922, 54.6902313
1: -25.3891716, 38.6557007, -17.9620037, 27.3516922, -52.7408638, 56.6177063
2: -25.1929913, 37.8162651, -17.7019806, 26.5621128, -51.7550964, 55.5182457
3: -31.9208298, 45.4162788, -22.8069592, 32.1879921, -64.1088257, 68.2232361
4: -28.5711136, 42.8956490, -20.1677132, 30.4125576, -58.9836731, 63.0633621

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113240
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4032405
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113240

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.3021946, 23.8024693, -37.1046600, 37.1046600
1: -15.2836475, 23.8005772, -15.2836475, 23.8005772, -39.0842171, 39.0842209
2: -15.1096706, 23.1136723, -15.1096706, 23.1136723, -38.2233429, 38.2233429
3: -19.4732094, 28.0227852, -19.4732094, 28.0227852, -47.4959869, 47.4959869
4: -17.4002285, 26.2846336, -17.4002285, 26.2846336, -43.6848602, 43.6848602

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3913150
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986158, upper bound: 46.3986158
time: 0.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -22.2130566, 39.0177612, -52.3199539, 46.0155220
1: -15.2836475, 23.8005772, -25.3891716, 38.6557007, -53.9393463, 49.1897392
2: -15.1096706, 23.1136723, -25.1929913, 37.8162651, -52.9259338, 48.3066559
3: -19.4732094, 28.0227852, -31.9208298, 45.4162788, -64.8894882, 59.9436111
4: -17.4002285, 26.2846336, -28.5711136, 42.8956490, -60.2958755, 54.8557472

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3938007
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986158, upper bound: 46.4011016
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -13.3021946, 23.8024693, -46.0155220, 52.3199539
1: -25.3891716, 38.6557007, -15.2836475, 23.8005772, -49.1897392, 53.9393463
2: -25.1929913, 37.8162651, -15.1096706, 23.1136723, -48.3066559, 52.9259338
3: -31.9208298, 45.4162788, -19.4732094, 28.0227852, -59.9436111, 64.8894806
4: -28.5711136, 42.8956490, -17.4002285, 26.2846336, -54.8557472, 60.2958755

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4022016, upper bound: 46.4074950
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989446, upper bound: 46.4079493
time: 0.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -22.2130566, 39.0177612, -61.2015877, 61.2015839
1: -25.3891716, 38.6557007, -25.3891716, 38.6557007, -64.0448761, 64.0448761
2: -25.1929913, 37.8162651, -25.1929913, 37.8162651, -62.9735107, 62.9735184
3: -31.9208298, 45.4162788, -31.9208298, 45.4162788, -77.3370972, 77.3370972
4: -28.5711136, 42.8956490, -28.5711136, 42.8956490, -71.4667664, 71.4667664

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4004123, upper bound: 46.4074950
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3913150
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.3986158, upper bound: 46.3986158
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3938007
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.3986158, upper bound: 46.4011016
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.4022016, upper bound: 46.4074950
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.3989446, upper bound: 46.4079493
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.4004123, upper bound: 46.4074950
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -13.3021946, 23.8024693, -36.1078033, 35.6570473
1: -14.1560287, 22.3223190, -15.2836475, 23.8005772, -37.9566040, 37.6059570
2: -14.0357513, 21.6778736, -15.1096706, 23.1136723, -37.1494217, 36.7875443
3: -18.0700207, 26.2806225, -19.4732094, 28.0227852, -46.0927963, 45.7538185
4: -16.2465935, 24.5797844, -17.4002285, 26.2846336, -42.5312271, 41.9800110

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3810521, upper bound: 46.3810521
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3810521, upper bound: 46.3913150
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -13.3021946, 23.8024693, -37.2860031, 37.4859200
1: -15.4869547, 24.1380043, -15.2836475, 23.8005772, -39.2875214, 39.4216461
2: -15.3245106, 23.4516239, -15.1096706, 23.1136723, -38.4381790, 38.5612946
3: -19.7292652, 28.4352989, -19.4732094, 28.0227852, -47.7520485, 47.9084969
4: -17.6341152, 26.6464577, -17.4002285, 26.2846336, -43.9187469, 44.0466843

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913150, upper bound: 46.3883529
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913150, upper bound: 46.3986158
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -22.2130566, 39.0177612, -51.3230934, 44.5679054
1: -14.1560287, 22.3223190, -25.3891716, 38.6557007, -52.8117294, 47.7114830
2: -14.0357513, 21.6778736, -25.1929913, 37.8162651, -51.8520164, 46.8708649
3: -18.0700207, 26.2806225, -31.9208298, 45.4162788, -63.4862938, 58.2014503
4: -16.2465935, 24.5797844, -28.5711136, 42.8956490, -59.1422424, 53.1508980

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913772, upper bound: 46.3936554
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3935789
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -22.2130566, 39.0177612, -52.5012894, 46.3967819
1: -15.4869547, 24.1380043, -25.3891716, 38.6557007, -54.1426544, 49.5271759
2: -15.3245106, 23.4516239, -25.1929913, 37.8162651, -53.1407700, 48.6446037
3: -19.7292652, 28.4352989, -31.9208298, 45.4162788, -65.1455460, 60.3561287
4: -17.6341152, 26.6464577, -28.5711136, 42.8956490, -60.5297623, 55.2175713

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033525, upper bound: 46.4000627
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3999862
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -13.3021946, 23.8024693, -45.3375244, 51.1967354
1: -24.6167240, 37.5179062, -15.2836475, 23.8005772, -48.4172974, 52.8015518
2: -24.4222260, 36.7149124, -15.1096706, 23.1136723, -47.5358963, 51.8245850
3: -30.9534435, 44.0686035, -19.4732094, 28.0227852, -58.9762230, 63.5418129
4: -27.6933746, 41.6349754, -17.4002285, 26.2846336, -53.9780083, 59.0352020

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3911697, upper bound: 46.3838657
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000627, upper bound: 46.4033525
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -13.3021946, 23.8024693, -47.0381165, 53.7481537
1: -26.5506458, 40.1011238, -15.2836475, 23.8005772, -50.3512230, 55.3847694
2: -26.2936344, 39.2237549, -15.1096706, 23.1136723, -49.4073067, 54.3334274
3: -33.3686562, 47.1178703, -19.4732094, 28.0227852, -61.3914337, 66.5910797
4: -29.7111588, 44.5652657, -17.4002285, 26.2846336, -55.9957924, 61.9654922

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3918314
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999862, upper bound: 46.4038066
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -22.2130566, 39.0177612, -60.5171471, 60.0526199
1: -24.6167240, 37.5179062, -25.3891716, 38.6557007, -63.2724228, 62.9070778
2: -24.4222260, 36.7149124, -25.1929913, 37.8162651, -62.2065315, 61.8460274
3: -30.9534435, 44.0686035, -31.9208298, 45.4162788, -76.3697205, 75.9894333
4: -27.6933746, 41.6349754, -28.5711136, 42.8956490, -70.5890198, 70.1991425

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4073164, upper bound: 46.4107520
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -22.2130566, 39.0177612, -62.2149239, 62.6284103
1: -26.5506458, 40.1011238, -25.3891716, 38.6557007, -65.2063446, 65.4902878
2: -26.2936344, 39.2237549, -25.1929913, 37.8162651, -64.0810699, 64.3736420
3: -33.3686562, 47.1178703, -31.9208298, 45.4162788, -78.7849197, 79.0386810
4: -29.7111588, 44.5652657, -28.5711136, 42.8956490, -72.6068115, 73.1325684

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4073164, upper bound: 46.4112063
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.26 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3810521, upper bound: 46.3810521
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3810521, upper bound: 46.3913150
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3913150, upper bound: 46.3883529
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3913150, upper bound: 46.3986158
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3913772, upper bound: 46.3936554
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3935789
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.4033525, upper bound: 46.4000627
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3999862
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3911697, upper bound: 46.3838657
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.4000627, upper bound: 46.4033525
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3918314
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3999862, upper bound: 46.4038066
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.4073164, upper bound: 46.4107520
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 3, lower bound: -46.4073164, upper bound: 46.4112063

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -12.3053322, 22.3548565, -34.6601868, 34.6601868
1: -14.1560287, 22.3223190, -14.1560287, 22.3223190, -36.4783478, 36.4783478
2: -14.0357513, 21.6778736, -14.0357513, 21.6778736, -35.7136230, 35.7136230
3: -18.0700207, 26.2806225, -18.0700207, 26.2806225, -44.3506355, 44.3506317
4: -16.2465935, 24.5797844, -16.2465935, 24.5797844, -40.8263779, 40.8263779

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3809068, upper bound: 46.3762536
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3764196
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -13.4835348, 24.1837368, -36.4890671, 35.8383865
1: -14.1560287, 22.3223190, -15.4869547, 24.1380043, -38.2940331, 37.8092651
2: -14.0357513, 21.6778736, -15.3245106, 23.4516239, -37.4873734, 37.0023842
3: -18.0700207, 26.2806225, -19.7292652, 28.4352989, -46.5053101, 46.0098839
4: -16.2465935, 24.5797844, -17.6341152, 26.6464577, -42.8930473, 42.2138977

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3809068, upper bound: 46.3882288
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3883948
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -12.3053322, 22.3548565, -35.8383865, 36.4890671
1: -15.4869547, 24.1380043, -14.1560287, 22.3223190, -37.8092690, 38.2940292
2: -15.3245106, 23.4516239, -14.0357513, 21.6778736, -37.0023842, 37.4873734
3: -19.7292652, 28.4352989, -18.0700207, 26.2806225, -46.0098801, 46.5053101
4: -17.6341152, 26.6464577, -16.2465935, 24.5797844, -42.2138977, 42.8930473

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3911696, upper bound: 46.3838657
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3828268
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -13.4835348, 24.1837368, -37.6672630, 37.6672630
1: -15.4869547, 24.1380043, -15.4869547, 24.1380043, -39.6249504, 39.6249542
2: -15.3245106, 23.4516239, -15.3245106, 23.4516239, -38.7761269, 38.7761269
3: -19.7292652, 28.4352989, -19.7292652, 28.4352989, -48.1645622, 48.1645622
4: -17.6341152, 26.6464577, -17.6341152, 26.6464577, -44.2805710, 44.2805710

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3911697, upper bound: 46.3958410
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3948021
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -21.5350533, 37.8945465, -50.1998787, 43.8899040
1: -14.1560287, 22.3223190, -24.6167240, 37.5179062, -51.6739349, 46.9390411
2: -14.0357513, 21.6778736, -24.4222260, 36.7149124, -50.7506638, 46.1000977
3: -18.0700207, 26.2806225, -30.9534435, 44.0686035, -62.1386185, 57.2340508
4: -16.2465935, 24.5797844, -27.6933746, 41.6349754, -57.8815651, 52.2731590

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3934129
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913772, upper bound: 46.3935789
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -23.2356472, 40.4459610, -52.7512932, 45.5904999
1: -14.1560287, 22.3223190, -26.5506458, 40.1011238, -54.2571526, 48.8729630
2: -14.0357513, 21.6778736, -26.2936344, 39.2237549, -53.2595062, 47.9715080
3: -18.0700207, 26.2806225, -33.3686562, 47.1178703, -65.1878738, 59.6492729
4: -16.2465935, 24.5797844, -29.7111588, 44.5652657, -60.8118591, 54.2909431

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3934129
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3935789
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -21.5350533, 37.8945465, -51.3780785, 45.7187843
1: -15.4869547, 24.1380043, -24.6167240, 37.5179062, -53.0048599, 48.7547226
2: -15.3245106, 23.4516239, -24.4222260, 36.7149124, -52.0394135, 47.8738480
3: -19.7292652, 28.4352989, -30.9534435, 44.0686035, -63.7978668, 59.3887329
4: -17.6341152, 26.6464577, -27.6933746, 41.6349754, -59.2690887, 54.3398323

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -23.2356472, 40.4459610, -53.9294968, 47.4193802
1: -15.4869547, 24.1380043, -26.5506458, 40.1011238, -55.5880775, 50.6886520
2: -15.3245106, 23.4516239, -26.2936344, 39.2237549, -54.5482635, 49.7452583
3: -19.7292652, 28.4352989, -33.3686562, 47.1178703, -66.8471375, 61.8039513
4: -17.6341152, 26.6464577, -29.7111588, 44.5652657, -62.1993790, 56.3576164

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -12.3053322, 22.3548565, -43.8899078, 50.1998787
1: -24.6167240, 37.5179062, -14.1560287, 22.3223190, -46.9390411, 51.6739349
2: -24.4222260, 36.7149124, -14.0357513, 21.6778736, -46.1000977, 50.7506638
3: -30.9534435, 44.0686035, -18.0700207, 26.2806225, -57.2340546, 62.1386185
4: -27.6933746, 41.6349754, -16.2465935, 24.5797844, -52.2731590, 57.8815651

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3911697, upper bound: 46.3833379
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3852423, upper bound: 46.3817303
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -13.4835348, 24.1837368, -45.7187843, 51.3780785
1: -24.6167240, 37.5179062, -15.4869547, 24.1380043, -48.7547264, 53.0048599
2: -24.4222260, 36.7149124, -15.3245106, 23.4516239, -47.8738480, 52.0394135
3: -30.9534435, 44.0686035, -19.7292652, 28.4352989, -59.3887329, 63.7978668
4: -27.6933746, 41.6349754, -17.6341152, 26.6464577, -54.3398323, 59.2690887

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953593, upper bound: 46.3999399
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -12.3053322, 22.3548565, -45.5905037, 52.7512932
1: -26.5506458, 40.1011238, -14.1560287, 22.3223190, -48.8729630, 54.2571526
2: -26.2936344, 39.2237549, -14.0357513, 21.6778736, -47.9715080, 53.2595062
3: -33.3686562, 47.1178703, -18.0700207, 26.2806225, -59.6492653, 65.1878815
4: -29.7111588, 44.5652657, -16.2465935, 24.5797844, -54.2909431, 60.8118591

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3935789, upper bound: 46.3913036
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3851399, upper bound: 46.3817278
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -13.4835348, 24.1837368, -47.4193802, 53.9294968
1: -26.5506458, 40.1011238, -15.4869547, 24.1380043, -50.6886520, 55.5880775
2: -26.2936344, 39.2237549, -15.3245106, 23.4516239, -49.7452583, 54.5482635
3: -33.3686562, 47.1178703, -19.7292652, 28.4352989, -61.8039513, 66.8471298
4: -29.7111588, 44.5652657, -17.6341152, 26.6464577, -56.3576164, 62.1993790

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005518
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -21.5350533, 37.8945465, -59.3681831, 59.3681831
1: -24.6167240, 37.5179062, -24.6167240, 37.5179062, -62.1346283, 62.1346283
2: -24.4222260, 36.7149124, -24.4222260, 36.7149124, -61.0790482, 61.0790482
3: -30.9534435, 44.0686035, -30.9534435, 44.0686035, -75.0220490, 75.0220490
4: -27.6933746, 41.6349754, -27.6933746, 41.6349754, -69.3153381, 69.3153381

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3998412, upper bound: 46.4067874
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -23.2356472, 40.4459610, -61.9439774, 61.0659599
1: -24.6167240, 37.5179062, -26.5506458, 40.1011238, -64.7178497, 64.0685501
2: -24.4222260, 36.7149124, -26.2936344, 39.2237549, -63.6066589, 62.9535904
3: -30.9534435, 44.0686035, -33.3686562, 47.1178703, -78.0713120, 77.4372559
4: -27.6933746, 41.6349754, -29.7111588, 44.5652657, -72.2487640, 71.3379593

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068585, upper bound: 46.4081789
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4080213
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -21.5350533, 37.8945465, -61.0659599, 61.9439774
1: -26.5506458, 40.1011238, -24.6167240, 37.5179062, -64.0685425, 64.7178497
2: -26.2936344, 39.2237549, -24.4222260, 36.7149124, -62.9535904, 63.6066589
3: -33.3686562, 47.1178703, -30.9534435, 44.0686035, -77.4372559, 78.0713120
4: -29.7111588, 44.5652657, -27.6933746, 41.6349754, -71.3379593, 72.2487564

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068584, upper bound: 46.4072416
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4050296, upper bound: 46.4072416
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -23.2356472, 40.4459610, -63.6417542, 63.6417503
1: -26.5506458, 40.1011238, -26.5506458, 40.1011238, -66.6517715, 66.6517715
2: -26.2936344, 39.2237549, -26.2936344, 39.2237549, -65.4812164, 65.4812088
3: -33.3686562, 47.1178703, -33.3686562, 47.1178703, -80.4865112, 80.4865112
4: -29.7111588, 44.5652657, -29.7111588, 44.5652657, -74.2713852, 74.2713852

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068585, upper bound: 46.4086331
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4050296, upper bound: 46.4086332
time: 1.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.54 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3809068, upper bound: 46.3762536
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3764196
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3809068, upper bound: 46.3882288
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3883948
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3911696, upper bound: 46.3838657
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3828268
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3911697, upper bound: 46.3958410
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3948021
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3934129
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3913772, upper bound: 46.3935789
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3934129
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3935789
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3911697, upper bound: 46.3833379
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3852423, upper bound: 46.3817303
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3953593, upper bound: 46.3999399
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3935789, upper bound: 46.3913036
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3851399, upper bound: 46.3817278
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005518
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.3998412, upper bound: 46.4067874
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4068585, upper bound: 46.4081789
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4080213
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4068584, upper bound: 46.4072416
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4050296, upper bound: 46.4072416
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4068585, upper bound: 46.4086331
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 3, lower bound: -46.4050296, upper bound: 46.4086332

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -12.3053322, 22.3548565, -34.0679588, 33.6229286
1: -13.4768963, 21.2664337, -14.1560287, 22.3223190, -35.7992134, 35.4224625
2: -13.3540964, 20.6594162, -14.0357513, 21.6778736, -35.0319710, 34.6951675
3: -17.2112961, 25.0253677, -18.0700207, 26.2806225, -43.4919167, 43.0953827
4: -15.4359884, 23.4152107, -16.2465935, 24.5797844, -40.0157738, 39.6618042

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -12.3053322, 22.3548565, -35.6521301, 36.0230942
1: -15.2800694, 23.7082329, -14.1560287, 22.3223190, -37.6023865, 37.8642616
2: -15.0858688, 23.0311298, -14.0357513, 21.6778736, -36.7637405, 37.0668793
3: -19.4841919, 27.9129372, -18.0700207, 26.2806225, -45.7648125, 45.9829445
4: -17.3320103, 26.2054291, -16.2465935, 24.5797844, -41.9117966, 42.4520187

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -13.4835348, 24.1837368, -35.8968391, 34.8011322
1: -13.4768963, 21.2664337, -15.4869547, 24.1380043, -37.6148987, 36.7533836
2: -13.3540964, 20.6594162, -15.3245106, 23.4516239, -36.8057175, 35.9839211
3: -17.2112961, 25.0253677, -19.7292652, 28.4352989, -45.6465950, 44.7546310
4: -15.4359884, 23.4152107, -17.6341152, 26.6464577, -42.0824432, 41.0493240

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -13.4835348, 24.1837368, -37.4810066, 37.2012978
1: -15.2800694, 23.7082329, -15.4869547, 24.1380043, -39.4180717, 39.1951866
2: -15.0858688, 23.0311298, -15.3245106, 23.4516239, -38.5374908, 38.3556366
3: -19.4841919, 27.9129372, -19.7292652, 28.4352989, -47.9194908, 47.6421967
4: -17.3320103, 26.2054291, -17.6341152, 26.6464577, -43.9784622, 43.8395462

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -12.3053322, 22.3548565, -35.2007637, 35.3926849
1: -14.7570267, 23.0229206, -14.1560287, 22.3223190, -37.0793457, 37.1789474
2: -14.5964413, 22.3779869, -14.0357513, 21.6778736, -36.2743149, 36.4137383
3: -18.8066769, 27.1103859, -18.0700207, 26.2806225, -45.0872879, 45.1803970
4: -16.7738895, 25.4232101, -16.2465935, 24.5797844, -41.3536758, 41.6697998

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -12.3053322, 22.3548565, -36.7887306, 37.7986755
1: -16.5651245, 25.4739952, -14.1560287, 22.3223190, -38.8874397, 39.6300240
2: -16.3321190, 24.7531395, -14.0357513, 21.6778736, -38.0099945, 38.7888832
3: -21.0962143, 30.0184078, -18.0700207, 26.2806225, -47.3768311, 48.0884171
4: -18.6865940, 28.2100029, -16.2465935, 24.5797844, -43.2663803, 44.4565964

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -13.4835348, 24.1837368, -37.0296402, 36.5708885
1: -14.7570267, 23.0229206, -15.4869547, 24.1380043, -38.8950310, 38.5098763
2: -14.5964413, 22.3779869, -15.3245106, 23.4516239, -38.0480652, 37.7024956
3: -18.8066769, 27.1103859, -19.7292652, 28.4352989, -47.2419624, 46.8396454
4: -16.7738895, 25.4232101, -17.6341152, 26.6464577, -43.4203491, 43.0573273

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3948020
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3944133, upper bound: 46.3948020
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -13.4835348, 24.1837368, -38.6176071, 38.9768753
1: -16.5651245, 25.4739952, -15.4869547, 24.1380043, -40.7031212, 40.9609489
2: -16.3321190, 24.7531395, -15.3245106, 23.4516239, -39.7837334, 40.0776329
3: -21.0962143, 30.0184078, -19.7292652, 28.4352989, -49.5315094, 49.7476616
4: -18.6865940, 28.2100029, -17.6341152, 26.6464577, -45.3330536, 45.8441162

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3948020
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3944132, upper bound: 46.3948020
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -21.5350533, 37.8945465, -49.6076508, 42.8526497
1: -13.4768963, 21.2664337, -24.6167240, 37.5179062, -50.9948044, 45.8831558
2: -13.3540964, 20.6594162, -24.4222260, 36.7149124, -50.0690079, 45.0816422
3: -17.2112961, 25.0253677, -30.9534435, 44.0686035, -61.2798996, 55.9788055
4: -15.4359884, 23.4152107, -27.6933746, 41.6349754, -57.0709610, 51.1085854

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3933564
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3892242
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -21.5350533, 37.8945465, -51.1918221, 45.2528152
1: -15.2800694, 23.7082329, -24.6167240, 37.5179062, -52.7979736, 48.3249550
2: -15.0858688, 23.0311298, -24.4222260, 36.7149124, -51.8007812, 47.4533539
3: -19.4841919, 27.9129372, -30.9534435, 44.0686035, -63.5527954, 58.8663712
4: -17.3320103, 26.2054291, -27.6933746, 41.6349754, -58.9669876, 53.8988037

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3833379, upper bound: 46.3936554
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895232
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -23.2356472, 40.4459610, -52.1590652, 44.5532455
1: -13.4768963, 21.2664337, -26.5506458, 40.1011238, -53.5780182, 47.8170776
2: -13.3540964, 20.6594162, -26.2936344, 39.2237549, -52.5778503, 46.9530449
3: -17.2112961, 25.0253677, -33.3686562, 47.1178703, -64.3291550, 58.3940163
4: -15.4359884, 23.4152107, -29.7111588, 44.5652657, -60.0012550, 53.1263695

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3932799
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894322
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -23.2356472, 40.4459610, -53.7432365, 46.9534073
1: -15.2800694, 23.7082329, -26.5506458, 40.1011238, -55.3811913, 50.2588806
2: -15.0858688, 23.0311298, -26.2936344, 39.2237549, -54.3096237, 49.3247643
3: -19.4841919, 27.9129372, -33.3686562, 47.1178703, -66.6020584, 61.2815857
4: -17.3320103, 26.2054291, -29.7111588, 44.5652657, -61.8972740, 55.9165878

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3935789
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -21.5350533, 37.8945465, -50.7404518, 44.6224060
1: -14.7570267, 23.0229206, -24.6167240, 37.5179062, -52.2749329, 47.6396446
2: -14.5964413, 22.3779869, -24.4222260, 36.7149124, -51.3113480, 46.8002129
3: -18.8066769, 27.1103859, -30.9534435, 44.0686035, -62.8752823, 58.0638275
4: -16.7738895, 25.4232101, -27.6933746, 41.6349754, -58.4088669, 53.1165848

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3953593
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -21.5350533, 37.8945465, -52.3284225, 47.0283966
1: -16.5651245, 25.4739952, -24.6167240, 37.5179062, -54.0830307, 50.0907211
2: -16.3321190, 24.7531395, -24.4222260, 36.7149124, -53.0470276, 49.1753578
3: -21.0962143, 30.0184078, -30.9534435, 44.0686035, -65.1648178, 60.9718437
4: -18.6865940, 28.2100029, -27.6933746, 41.6349754, -60.3215675, 55.9033775

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -23.2356472, 40.4459610, -53.2918701, 46.3229980
1: -14.7570267, 23.0229206, -26.5506458, 40.1011238, -54.8581505, 49.5735664
2: -14.5964413, 22.3779869, -26.2936344, 39.2237549, -53.8201981, 48.6716194
3: -18.8066769, 27.1103859, -33.3686562, 47.1178703, -65.9245453, 60.4790344
4: -16.7738895, 25.4232101, -29.7111588, 44.5652657, -61.3391571, 55.1343689

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000976, upper bound: 46.3994150
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3955673
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -23.2356472, 40.4459610, -54.8798409, 48.7289886
1: -16.5651245, 25.4739952, -26.5506458, 40.1011238, -56.6662445, 52.0246429
2: -16.3321190, 24.7531395, -26.2936344, 39.2237549, -55.5558739, 51.0467644
3: -21.0962143, 30.0184078, -33.3686562, 47.1178703, -68.2140808, 63.3870544
4: -18.6865940, 28.2100029, -29.7111588, 44.5652657, -63.2518616, 57.9211617

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994150
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3954985
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -12.3053322, 22.3548565, -43.8367081, 50.1226082
1: -24.5586433, 37.5217667, -14.1560287, 22.3223190, -46.8809624, 51.6777954
2: -24.4099751, 36.6918221, -14.0357513, 21.6778736, -46.0878487, 50.7275734
3: -30.8934937, 44.0957451, -18.0700207, 26.2806225, -57.1741180, 62.1657562
4: -27.7538719, 41.6327591, -16.2465935, 24.5797844, -52.3336563, 57.8793526

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908707, upper bound: 46.3908494
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908707, upper bound: 46.3833379
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -12.3053322, 22.3548565, -43.6204681, 49.7724304
1: -24.3087234, 37.0905914, -14.1560287, 22.3223190, -46.6310387, 51.2466202
2: -24.1263924, 36.2956047, -14.0357513, 21.6778736, -45.8042679, 50.3313560
3: -30.5667114, 43.5680695, -18.0700207, 26.2806225, -56.8473244, 61.6380806
4: -27.3722706, 41.1605835, -16.2465935, 24.5797844, -51.9520569, 57.4071770

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3849433, upper bound: 46.3906917
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892242, upper bound: 46.3906917
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -13.4835348, 24.1837368, -45.6655846, 51.3008118
1: -24.5586433, 37.5217667, -15.4869547, 24.1380043, -48.6966476, 53.0087204
2: -24.4099751, 36.6918221, -15.3245106, 23.4516239, -47.8615990, 52.0163231
3: -30.8934937, 44.0957451, -19.7292652, 28.4352989, -59.3287926, 63.8250122
4: -27.7538719, 41.6327591, -17.6341152, 26.6464577, -54.4003296, 59.2668762

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3970057, upper bound: 46.3925860
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -13.4835348, 24.1837368, -45.4493484, 50.9506340
1: -24.3087234, 37.0905914, -15.4869547, 24.1380043, -48.4467201, 52.5775452
2: -24.1263924, 36.2956047, -15.3245106, 23.4516239, -47.5780144, 51.6201057
3: -30.5667114, 43.5680695, -19.7292652, 28.4352989, -59.0020103, 63.2973328
4: -27.3722706, 41.1605835, -17.6341152, 26.6464577, -54.0187263, 58.7947006

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3999399
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3999399
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.1895599, 40.3705215, -12.3053322, 22.3548565, -45.5444069, 52.6758537
1: -26.5017738, 40.0988922, -14.1560287, 22.3223190, -48.8240929, 54.2549210
2: -26.2765026, 39.1973801, -14.0357513, 21.6778736, -47.9543762, 53.2331314
3: -33.3155136, 47.1392517, -18.0700207, 26.2806225, -59.5961304, 65.2092743
4: -29.7744904, 44.5557632, -16.2465935, 24.5797844, -54.3542709, 60.8023567

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880959, upper bound: 46.3822990
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932799, upper bound: 46.3913036
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.9565926, 40.0076714, -12.3053322, 22.3548565, -45.3114433, 52.3130035
1: -26.2316208, 39.6613235, -14.1560287, 22.3223190, -48.5539398, 53.8173523
2: -25.9845924, 38.7927017, -14.0357513, 21.6778736, -47.6624680, 52.8284531
3: -32.9699440, 46.6054611, -18.0700207, 26.2806225, -59.2505608, 64.6754837
4: -29.3801270, 44.0779724, -16.2465935, 24.5797844, -53.9599075, 60.3245659

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3913036
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3894322, upper bound: 46.3913036
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.1895599, 40.3705215, -13.4835348, 24.1837368, -47.3732872, 53.8540535
1: -26.5017738, 40.0988922, -15.4869547, 24.1380043, -50.6397781, 55.5858459
2: -26.2765026, 39.1973801, -15.3245106, 23.4516239, -49.7281265, 54.5218811
3: -33.3155136, 47.1392517, -19.7292652, 28.4352989, -61.7508125, 66.8685150
4: -29.7744904, 44.5557632, -17.6341152, 26.6464577, -56.4209442, 62.1898804

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.9565926, 40.0076714, -13.4835348, 24.1837368, -47.1403236, 53.4911995
1: -26.2316208, 39.6613235, -15.4869547, 24.1380043, -50.3696251, 55.1482773
2: -25.9845924, 38.7927017, -15.3245106, 23.4516239, -49.4362183, 54.1172028
3: -32.9699440, 46.6054611, -19.7292652, 28.4352989, -61.4052429, 66.3347244
4: -29.3801270, 44.0779724, -17.6341152, 26.6464577, -56.0265732, 61.7120895

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005517
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005518
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -21.5350533, 37.8945465, -59.3369331, 59.2810364
1: -24.5586433, 37.5217667, -24.6167240, 37.5179062, -62.0765495, 62.1384888
2: -24.4099751, 36.6918221, -24.4222260, 36.7149124, -61.0640945, 61.0488091
3: -30.8934937, 44.0957451, -30.9534435, 44.0686035, -74.9620895, 75.0491867
4: -27.7538719, 41.6327591, -27.6933746, 41.6349754, -69.3829041, 69.3211899

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -21.5350533, 37.8945465, -59.0963364, 58.9460907
1: -24.3087234, 37.0905914, -24.6167240, 37.5179062, -61.8266296, 61.7073135
2: -24.1263924, 36.2956047, -24.4222260, 36.7149124, -60.7806091, 60.6627350
3: -30.5667114, 43.5680695, -30.9534435, 44.0686035, -74.6353149, 74.5215149
4: -27.3722706, 41.1605835, -27.6933746, 41.6349754, -68.9950638, 68.8385544

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953825, upper bound: 46.3953825
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -23.2356472, 40.4459610, -61.9127312, 60.9788094
1: -24.5586433, 37.5217667, -26.5506458, 40.1011238, -64.6597672, 64.0724106
2: -24.4099751, 36.6918221, -26.2936344, 39.2237549, -63.5917130, 62.9233551
3: -30.8934937, 44.0957451, -33.3686562, 47.1178703, -78.0113525, 77.4644012
4: -27.7538719, 41.6327591, -29.7111588, 44.5652657, -72.3163300, 71.3438110

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.3953825
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -23.2356472, 40.4459610, -61.6721306, 60.6438713
1: -24.3087234, 37.0905914, -26.5506458, 40.1011238, -64.4098434, 63.6412354
2: -24.1263924, 36.2956047, -26.2936344, 39.2237549, -63.3082123, 62.5372772
3: -30.5667114, 43.5680695, -33.3686562, 47.1178703, -77.6845779, 76.9367218
4: -27.3722706, 41.1605835, -29.7111588, 44.5652657, -71.9284821, 70.8611832

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.3953825
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4080213
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.1895599, 40.3705215, -21.5350533, 37.8945465, -61.0422020, 61.8612099
1: -26.5017738, 40.0988922, -24.6167240, 37.5179062, -64.0196686, 64.7156143
2: -26.2765026, 39.1973801, -24.4222260, 36.7149124, -62.9456100, 63.5758095
3: -33.3155136, 47.1392517, -30.9534435, 44.0686035, -77.3841171, 78.0926971
4: -29.7744904, 44.5557632, -27.6933746, 41.6349754, -71.4094620, 72.2491302

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953825, upper bound: 46.4072417
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953825, upper bound: 46.4072416
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.9565926, 40.0076714, -21.5350533, 37.8945465, -60.7845306, 61.5111504
1: -26.2316208, 39.6613235, -24.6167240, 37.5179062, -63.7495270, 64.2780457
2: -25.9845924, 38.7927017, -24.4222260, 36.7149124, -62.6443901, 63.1791954
3: -32.9699440, 46.6054611, -30.9534435, 44.0686035, -77.0385437, 77.5589066
4: -29.3801270, 44.0779724, -27.6933746, 41.6349754, -71.0078583, 71.7591553

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4050296, upper bound: 46.4072417
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953825, upper bound: 46.4072417
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.1895599, 40.3705215, -23.2356472, 40.4459610, -63.6180038, 63.5589905
1: -26.5017738, 40.0988922, -26.5506458, 40.1011238, -66.6028900, 66.6495361
2: -26.2765026, 39.1973801, -26.2936344, 39.2237549, -65.4732208, 65.4503555
3: -33.3155136, 47.1392517, -33.3686562, 47.1178703, -80.4333801, 80.5079041
4: -29.7744904, 44.5557632, -29.7111588, 44.5652657, -74.3397446, 74.2669220

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4086331
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4055103, upper bound: 46.4086331
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.9565926, 40.0076714, -23.2356472, 40.4459610, -63.3603287, 63.2089310
1: -26.2316208, 39.6613235, -26.5506458, 40.1011238, -66.3327332, 66.2119675
2: -25.9845924, 38.7927017, -26.2936344, 39.2237549, -65.1720047, 65.0537415
3: -32.9699440, 46.6054611, -33.3686562, 47.1178703, -80.0877991, 79.9741211
4: -29.3801270, 44.0779724, -29.7111588, 44.5652657, -73.9412918, 73.7817688

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4055103, upper bound: 46.4086332
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.3953800
time: 0.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3948020
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3944133, upper bound: 46.3948020
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3948020
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3944132, upper bound: 46.3948020
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3933564
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3892242
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3833379, upper bound: 46.3936554
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895232
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3932799
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894322
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3935789
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3953593
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4000976, upper bound: 46.3994150
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3955673
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994150
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3954985
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3908707, upper bound: 46.3908494
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3908707, upper bound: 46.3833379
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3849433, upper bound: 46.3906917
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3892242, upper bound: 46.3906917
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3970057, upper bound: 46.3925860
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3999399
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3999399
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3880959, upper bound: 46.3822990
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3932799, upper bound: 46.3913036
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3913036
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3894322, upper bound: 46.3913036
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005517
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005518
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4028929, upper bound: 46.4066297
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953825, upper bound: 46.3953825
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953800, upper bound: 46.3953825
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953800, upper bound: 46.3953825
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4080213
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953825, upper bound: 46.4072417
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953825, upper bound: 46.4072416
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4050296, upper bound: 46.4072417
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953825, upper bound: 46.4072417
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4086331
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4055103, upper bound: 46.4086331
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.4055103, upper bound: 46.4086332
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -46.3953800, upper bound: 46.3953800

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -11.7131090, 21.3175964, -33.0307045, 33.0307045
1: -13.4768963, 21.2664337, -13.4768963, 21.2664337, -34.7433319, 34.7433319
2: -13.3540964, 20.6594162, -13.3540964, 20.6594162, -34.0135117, 34.0135117
3: -17.2112961, 25.0253677, -17.2112961, 25.0253677, -42.2366638, 42.2366638
4: -15.4359884, 23.4152107, -15.4359884, 23.4152107, -38.8512001, 38.8512001

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806078, upper bound: 46.3757257
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3755927
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -13.2972765, 23.7177620, -35.4308701, 34.6148720
1: -13.4768963, 21.2664337, -15.2800694, 23.7082329, -37.1851273, 36.5465012
2: -13.3540964, 20.6594162, -15.0858688, 23.0311298, -36.3852272, 35.7452850
3: -17.2112961, 25.0253677, -19.4841919, 27.9129372, -45.1242294, 44.5095596
4: -15.4359884, 23.4152107, -17.3320103, 26.2054291, -41.6414146, 40.7472229

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806078, upper bound: 46.3757257
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3755927
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -11.7131090, 21.3175964, -34.6148720, 35.4308701
1: -15.2800694, 23.7082329, -13.4768963, 21.2664337, -36.5465012, 37.1851273
2: -15.0858688, 23.0311298, -13.3540964, 20.6594162, -35.7452850, 36.3852272
3: -19.4841919, 27.9129372, -17.2112961, 25.0253677, -44.5095596, 45.1242294
4: -17.3320103, 26.2054291, -15.4359884, 23.4152107, -40.7472229, 41.6414146

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3761206, upper bound: 46.3757726
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3758917
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -13.2972765, 23.7177620, -37.0150375, 37.0150375
1: -15.2800694, 23.7082329, -15.2800694, 23.7082329, -38.9883041, 38.9883003
2: -15.0858688, 23.0311298, -15.0858688, 23.0311298, -38.1169968, 38.1169968
3: -19.4841919, 27.9129372, -19.4841919, 27.9129372, -47.3971291, 47.3971291
4: -17.3320103, 26.2054291, -17.3320103, 26.2054291, -43.5374336, 43.5374374

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3761206, upper bound: 46.3757726
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3758917
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -12.8459072, 23.0873528, -34.8004608, 34.1635056
1: -13.4768963, 21.2664337, -14.7570267, 23.0229206, -36.4998169, 36.0234604
2: -13.3540964, 20.6594162, -14.5964413, 22.3779869, -35.7320824, 35.2558594
3: -17.2112961, 25.0253677, -18.8066769, 27.1103859, -44.3216782, 43.8320389
4: -15.4359884, 23.4152107, -16.7738895, 25.4232101, -40.8591995, 40.1891022

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3867428, upper bound: 46.3849739
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3848409
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -14.4338799, 25.4933434, -37.2064476, 35.7514763
1: -13.4768963, 21.2664337, -16.5651245, 25.4739952, -38.9508896, 37.8315582
2: -13.3540964, 20.6594162, -16.3321190, 24.7531395, -38.1072311, 36.9915237
3: -17.2112961, 25.0253677, -21.0962143, 30.0184078, -47.2296944, 46.1215820
4: -15.4359884, 23.4152107, -18.6865940, 28.2100029, -43.6459923, 42.1018066

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3867428, upper bound: 46.3849739
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3848409
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -12.8459072, 23.0873528, -36.3846283, 36.5636673
1: -15.2800694, 23.7082329, -14.7570267, 23.0229206, -38.3029900, 38.4652596
2: -15.0858688, 23.0311298, -14.5964413, 22.3779869, -37.4638557, 37.6275711
3: -19.4841919, 27.9129372, -18.8066769, 27.1103859, -46.5945778, 46.7196007
4: -17.3320103, 26.2054291, -16.7738895, 25.4232101, -42.7552185, 42.9793167

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3822556, upper bound: 46.3850207
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3851399
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -14.4338799, 25.4933434, -38.7906189, 38.1516418
1: -15.2800694, 23.7082329, -16.5651245, 25.4739952, -40.7540665, 40.2733574
2: -15.0858688, 23.0311298, -16.3321190, 24.7531395, -39.8389931, 39.3632431
3: -19.4841919, 27.9129372, -21.0962143, 30.0184078, -49.5025978, 49.0091438
4: -17.3320103, 26.2054291, -18.6865940, 28.2100029, -45.5420113, 44.8920212

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3761206, upper bound: 46.3850207
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3851399
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -11.7131090, 21.3175964, -34.1635056, 34.8004570
1: -14.7570267, 23.0229206, -13.4768963, 21.2664337, -36.0234604, 36.4998169
2: -14.5964413, 22.3779869, -13.3540964, 20.6594162, -35.2558594, 35.7320824
3: -18.8066769, 27.1103859, -17.2112961, 25.0253677, -43.8320351, 44.3216820
4: -16.7738895, 25.4232101, -15.4359884, 23.4152107, -40.1891022, 40.8591995

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3698143, upper bound: 46.3715566
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892987, upper bound: 46.3813073
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -13.2972765, 23.7177620, -36.5636673, 36.3846283
1: -14.7570267, 23.0229206, -15.2800694, 23.7082329, -38.4652596, 38.3029900
2: -14.5964413, 22.3779869, -15.0858688, 23.0311298, -37.6275711, 37.4638557
3: -18.8066769, 27.1103859, -19.4841919, 27.9129372, -46.7196007, 46.5945778
4: -16.7738895, 25.4232101, -17.3320103, 26.2054291, -42.9793167, 42.7552185

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3698143, upper bound: 46.3715566
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892987, upper bound: 46.3813073
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -11.7131090, 21.3175964, -35.7514763, 37.2064476
1: -16.5651245, 25.4739952, -13.4768963, 21.2664337, -37.8315582, 38.9508896
2: -16.3321190, 24.7531395, -13.3540964, 20.6594162, -36.9915237, 38.1072235
3: -21.0962143, 30.0184078, -17.2112961, 25.0253677, -46.1215782, 47.2296944
4: -18.6865940, 28.2100029, -15.4359884, 23.4152107, -42.1018066, 43.6459923

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880958, upper bound: 46.3822990
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3817278
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -13.2972765, 23.7177620, -38.1516418, 38.7906189
1: -16.5651245, 25.4739952, -15.2800694, 23.7082329, -40.2733536, 40.7540665
2: -16.3321190, 24.7531395, -15.0858688, 23.0311298, -39.3632431, 39.8389969
3: -21.0962143, 30.0184078, -19.4841919, 27.9129372, -49.0091438, 49.5026016
4: -18.6865940, 28.2100029, -17.3320103, 26.2054291, -44.8920212, 45.5420113

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880958, upper bound: 46.3822990
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3817278
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -12.8459072, 23.0873528, -35.9332581, 35.9332581
1: -14.7570267, 23.0229206, -14.7570267, 23.0229206, -37.7799454, 37.7799454
2: -14.5964413, 22.3779869, -14.5964413, 22.3779869, -36.9744263, 36.9744263
3: -18.8066769, 27.1103859, -18.8066769, 27.1103859, -45.9170532, 45.9170532
4: -16.7738895, 25.4232101, -16.7738895, 25.4232101, -42.1970978, 42.1970978

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3738053, upper bound: 46.3807300
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3960314, upper bound: 46.3941067
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -14.4338799, 25.4933434, -38.3392487, 37.5212288
1: -14.7570267, 23.0229206, -16.5651245, 25.4739952, -40.2310219, 39.5880432
2: -14.5964413, 22.3779869, -16.3321190, 24.7531395, -39.3495712, 38.7101021
3: -18.8066769, 27.1103859, -21.0962143, 30.0184078, -48.8250732, 48.2065926
4: -16.7738895, 25.4232101, -18.6865940, 28.2100029, -44.9838943, 44.1098022

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3738053, upper bound: 46.3807300
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3960314, upper bound: 46.3941067
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -12.8459072, 23.0873528, -37.5212288, 38.3392487
1: -16.5651245, 25.4739952, -14.7570267, 23.0229206, -39.5880432, 40.2310219
2: -16.3321190, 24.7531395, -14.5964413, 22.3779869, -38.7101021, 39.3495712
3: -21.0962143, 30.0184078, -18.8066769, 27.1103859, -48.2065964, 48.8250732
4: -18.6865940, 28.2100029, -16.7738895, 25.4232101, -44.1098022, 44.9838943

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3937827, upper bound: 46.3915471
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908924, upper bound: 46.3909759
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -14.4338799, 25.4933434, -39.9272194, 39.9272194
1: -16.5651245, 25.4739952, -16.5651245, 25.4739952, -42.0391197, 42.0391197
2: -16.3321190, 24.7531395, -16.3321190, 24.7531395, -41.0852432, 41.0852394
3: -21.0962143, 30.0184078, -21.0962143, 30.0184078, -51.1146164, 51.1146164
4: -18.6865940, 28.2100029, -18.6865940, 28.2100029, -46.8965988, 46.8965988

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3937827, upper bound: 46.3915471
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908923, upper bound: 46.3909759
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -21.4818611, 37.8172760, -49.5303841, 42.7994537
1: -13.4768963, 21.2664337, -24.5586433, 37.5217667, -50.9986649, 45.8250771
2: -13.3540964, 20.6594162, -24.4099751, 36.6918221, -50.0459175, 45.0693855
3: -17.2112961, 25.0253677, -30.8934937, 44.0957451, -61.3070412, 55.9188614
4: -15.4359884, 23.4152107, -27.7538719, 41.6327591, -57.0687485, 51.1690826

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3892242
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3892242
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -21.2656136, 37.4670982, -49.1802025, 42.5832100
1: -13.4768963, 21.2664337, -24.3087234, 37.0905914, -50.5674896, 45.5751572
2: -13.3540964, 20.6594162, -24.1263924, 36.2956047, -49.6496964, 44.7858047
3: -17.2112961, 25.0253677, -30.5667114, 43.5680695, -60.7793655, 55.5920753
4: -15.4359884, 23.4152107, -27.3722706, 41.1605835, -56.5965729, 50.7874832

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3892242
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3892242
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -21.4818611, 37.8172760, -51.1145515, 45.1996155
1: -15.2800694, 23.7082329, -24.5586433, 37.5217667, -52.8018341, 48.2668762
2: -15.0858688, 23.0311298, -24.4099751, 36.6918221, -51.7776909, 47.4411049
3: -19.4841919, 27.9129372, -30.8934937, 44.0957451, -63.5799370, 58.8064308
4: -17.3320103, 26.2054291, -27.7538719, 41.6327591, -58.9647675, 53.9593010

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3894040
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3895232
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -21.2656136, 37.4670982, -50.7643738, 44.9833755
1: -15.2800694, 23.7082329, -24.3087234, 37.0905914, -52.3706589, 48.0169525
2: -15.0858688, 23.0311298, -24.1263924, 36.2956047, -51.3814697, 47.1575241
3: -19.4841919, 27.9129372, -30.5667114, 43.5680695, -63.0522614, 58.4796410
4: -17.3320103, 26.2054291, -27.3722706, 41.1605835, -58.4925919, 53.5776978

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894040
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3895231
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -23.1895599, 40.3705215, -52.0836258, 44.5071564
1: -13.4768963, 21.2664337, -26.5017738, 40.0988922, -53.5757904, 47.7682076
2: -13.3540964, 20.6594162, -26.2765026, 39.1973801, -52.5514755, 46.9359169
3: -17.2112961, 25.0253677, -33.3155136, 47.1392517, -64.3505478, 58.3408775
4: -15.4359884, 23.4152107, -29.7744904, 44.5557632, -59.9917526, 53.1896973

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3894322
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913036, upper bound: 46.3894322
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -22.9565926, 40.0076714, -51.7207718, 44.2741890
1: -13.4768963, 21.2664337, -26.2316208, 39.6613235, -53.1382217, 47.4980545
2: -13.3540964, 20.6594162, -25.9845924, 38.7927017, -52.1467972, 46.6440048
3: -17.2112961, 25.0253677, -32.9699440, 46.6054611, -63.8167534, 57.9953117
4: -15.4359884, 23.4152107, -29.3801270, 44.0779724, -59.5139618, 52.7953300

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3894322
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3894322
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -23.1895599, 40.3705215, -53.6677971, 46.9073219
1: -15.2800694, 23.7082329, -26.5017738, 40.0988922, -55.3789597, 50.2100067
2: -15.0858688, 23.0311298, -26.2765026, 39.1973801, -54.2832451, 49.3076324
3: -19.4841919, 27.9129372, -33.3155136, 47.1392517, -66.6234436, 61.2284393
4: -17.3320103, 26.2054291, -29.7744904, 44.5557632, -61.8877716, 55.9799118

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895200
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -22.9565926, 40.0076714, -53.3049469, 46.6743546
1: -15.2800694, 23.7082329, -26.2316208, 39.6613235, -54.9413910, 49.9398537
2: -15.0858688, 23.0311298, -25.9845924, 38.7927017, -53.8785667, 49.0157242
3: -19.4841919, 27.9129372, -32.9699440, 46.6054611, -66.0896530, 60.8828812
4: -17.3320103, 26.2054291, -29.3801270, 44.0779724, -61.4099808, 55.5855446

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3895200
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3897312
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -21.4818611, 37.8172760, -50.6631851, 44.5692062
1: -14.7570267, 23.0229206, -24.5586433, 37.5217667, -52.2787933, 47.5815659
2: -14.5964413, 22.3779869, -24.4099751, 36.6918221, -51.2882576, 46.7879639
3: -18.8066769, 27.1103859, -30.8934937, 44.0957451, -62.9024200, 58.0038795
4: -16.7738895, 25.4232101, -27.7538719, 41.6327591, -58.4066467, 53.1770821

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3843470, upper bound: 46.3904746
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986879, upper bound: 46.3986448
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -21.2656136, 37.4670982, -50.3130035, 44.3529663
1: -14.7570267, 23.0229206, -24.3087234, 37.0905914, -51.8476181, 47.3316422
2: -14.5964413, 22.3779869, -24.1263924, 36.2956047, -50.8920441, 46.5043793
3: -18.8066769, 27.1103859, -30.5667114, 43.5680695, -62.3747444, 57.6770973
4: -16.7738895, 25.4232101, -27.3722706, 41.1605835, -57.9344711, 52.7954788

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3841693, upper bound: 46.3862507
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985102, upper bound: 46.3944209
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -21.4818611, 37.8172760, -52.2511559, 46.9751968
1: -16.5651245, 25.4739952, -24.5586433, 37.5217667, -54.0868912, 50.0326385
2: -16.3321190, 24.7531395, -24.4099751, 36.6918221, -53.0239372, 49.1631088
3: -21.0962143, 30.0184078, -30.8934937, 44.0957451, -65.1919556, 60.9119034
4: -18.6865940, 28.2100029, -27.7538719, 41.6327591, -60.3193512, 55.9638748

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3953593
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -21.2656136, 37.4670982, -51.9009781, 46.7589569
1: -16.5651245, 25.4739952, -24.3087234, 37.0905914, -53.6557159, 49.7827187
2: -16.3321190, 24.7531395, -24.1263924, 36.2956047, -52.6277161, 48.8795242
3: -21.0962143, 30.0184078, -30.5667114, 43.5680695, -64.6642761, 60.5851059
4: -18.6865940, 28.2100029, -27.3722706, 41.1605835, -59.8471756, 55.5822754

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3953593
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3909785, upper bound: 46.3953593
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -23.1895599, 40.3705215, -53.2164268, 46.2769051
1: -14.7570267, 23.0229206, -26.5017738, 40.0988922, -54.8559189, 49.5246964
2: -14.5964413, 22.3779869, -26.2765026, 39.1973801, -53.7938156, 48.6544876
3: -18.8066769, 27.1103859, -33.3155136, 47.1392517, -65.9459305, 60.4258881
4: -16.7738895, 25.4232101, -29.7744904, 44.5557632, -61.3296509, 55.1976967

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755345, upper bound: 46.3904746
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990836, upper bound: 46.3986448
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -22.9565926, 40.0076714, -52.8535767, 46.0439453
1: -14.7570267, 23.0229206, -26.2316208, 39.6613235, -54.4183502, 49.2545395
2: -14.5964413, 22.3779869, -25.9845924, 38.7927017, -53.3891411, 48.3625793
3: -18.8066769, 27.1103859, -32.9699440, 46.6054611, -65.4121399, 60.0803299
4: -16.7738895, 25.4232101, -29.3801270, 44.0779724, -60.8518600, 54.8033295

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3847427, upper bound: 46.3865946
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990836, upper bound: 46.3947648
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -23.1895599, 40.3705215, -54.8043976, 48.6828995
1: -16.5651245, 25.4739952, -26.5017738, 40.0988922, -56.6640167, 51.9757690
2: -16.3321190, 24.7531395, -26.2765026, 39.1973801, -55.5294952, 51.0296326
3: -21.0962143, 30.0184078, -33.3155136, 47.1392517, -68.2354660, 63.3339195
4: -18.6865940, 28.2100029, -29.7744904, 44.5557632, -63.2423553, 57.9844894

Time for backsubstitution: 2.71 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
time: 0.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -15.6748066, 27.2767467, -40.5789375, 39.4772720
1: -15.2836475, 23.8005772, -17.9646091, 27.3548412, -42.6384850, 41.7651787
2: -15.1096706, 23.1136723, -17.7045784, 26.5651569, -41.6748276, 40.8182526
3: -19.4732094, 28.0227852, -22.8100491, 32.1917152, -51.6649170, 50.8328323
4: -17.4002285, 26.2846336, -20.1704769, 30.4162006, -47.8164215, 46.4551048

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
time: 0.79 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -15.6405516, 27.2283001, -49.4413528, 54.6583061
1: -25.3891716, 38.6557007, -17.9264469, 27.3087788, -52.6979446, 56.5821457
2: -25.1929913, 37.8162651, -17.6665192, 26.5205688, -51.7135582, 55.4827843
3: -31.9208298, 45.4162788, -22.7647934, 32.1371918, -64.0580215, 68.1810760
4: -28.5711136, 42.8956490, -20.1301193, 30.3629417, -58.9340553, 63.0257568

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113062
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113062

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.3021946, 23.8024693, -37.1046600, 37.1046600
1: -15.2836475, 23.8005772, -15.2836475, 23.8005772, -39.0842171, 39.0842209
2: -15.1096706, 23.1136723, -15.1096706, 23.1136723, -38.2233429, 38.2233429
3: -19.4732094, 28.0227852, -19.4732094, 28.0227852, -47.4959869, 47.4959869
4: -17.4002285, 26.2846336, -17.4002285, 26.2846336, -43.6848602, 43.6848602

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3912553
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986153, upper bound: 46.3986153
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -22.2130566, 39.0177612, -52.3199539, 46.0155220
1: -15.2836475, 23.8005772, -25.3891716, 38.6557007, -53.9393463, 49.1897392
2: -15.1096706, 23.1136723, -25.1929913, 37.8162651, -52.9259338, 48.3066559
3: -19.4732094, 28.0227852, -31.9208298, 45.4162788, -64.8894882, 59.9436111
4: -17.4002285, 26.2846336, -28.5711136, 42.8956490, -60.2958755, 54.8557472

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3937360
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986153, upper bound: 46.4011016
time: 0.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -13.3021946, 23.8024693, -46.0155220, 52.3199539
1: -25.3891716, 38.6557007, -15.2836475, 23.8005772, -49.1897392, 53.9393463
2: -25.1929913, 37.8162651, -15.1096706, 23.1136723, -48.3066559, 52.9259338
3: -31.9208298, 45.4162788, -19.4732094, 28.0227852, -59.9436111, 64.8894806
4: -28.5711136, 42.8956490, -17.4002285, 26.2846336, -54.8557472, 60.2958755

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4004123, upper bound: 46.4074950
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989446, upper bound: 46.4079492
time: 1.00 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -22.2130566, 39.0177612, -61.2015877, 61.2015839
1: -25.3891716, 38.6557007, -25.3891716, 38.6557007, -64.0448761, 64.0448761
2: -25.1929913, 37.8162651, -25.1929913, 37.8162651, -62.9735107, 62.9735184
3: -31.9208298, 45.4162788, -31.9208298, 45.4162788, -77.3370972, 77.3370972
4: -28.5711136, 42.8956490, -28.5711136, 42.8956490, -71.4667664, 71.4667664

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4004123, upper bound: 46.3999835
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.16 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3912553
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.3986153, upper bound: 46.3986153
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.3883529, upper bound: 46.3937360
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.3986153, upper bound: 46.4011016
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.4004123, upper bound: 46.4074950
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.3989446, upper bound: 46.4079492
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.4004123, upper bound: 46.3999835
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -46.3989446, upper bound: 46.3989446

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -13.2302017, 23.6994343, -36.0047684, 35.5850525
1: -14.1560287, 22.3223190, -15.2028027, 23.6951675, -37.8511963, 37.5251198
2: -14.0357513, 21.6778736, -15.0325298, 23.0114632, -37.0472145, 36.7104034
3: -18.0700207, 26.2806225, -19.3725147, 27.8980236, -45.9680367, 45.6531334
4: -16.2465935, 24.5797844, -17.3170547, 26.1628933, -42.4094772, 41.8968391

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3810150, upper bound: 46.3810150
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3810150, upper bound: 46.3912553
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -13.2225647, 23.6556568, -37.1391907, 37.4062996
1: -15.4869547, 24.1380043, -15.1915722, 23.6609211, -39.1478767, 39.3295670
2: -15.3245106, 23.4516239, -15.0139675, 22.9749470, -38.2994499, 38.4655876
3: -19.7292652, 28.4352989, -19.3571091, 27.8597507, -47.5890121, 47.7924042
4: -17.6341152, 26.6464577, -17.2891045, 26.1276150, -43.7617302, 43.9355621

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3837861, upper bound: 46.3975769
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -22.1379128, 38.9102364, -51.2155685, 44.4927597
1: -14.1560287, 22.3223190, -25.3038750, 38.5469818, -52.7030106, 47.6261864
2: -14.0357513, 21.6778736, -25.1110973, 37.7102280, -51.7459793, 46.7889709
3: -18.0700207, 26.2806225, -31.8146172, 45.2882767, -63.3582916, 58.0952301
4: -16.2465935, 24.5797844, -28.4848862, 42.7703819, -59.0169754, 53.0646667

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913772, upper bound: 46.3936554
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3935789
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -22.0881176, 38.8044281, -52.2879524, 46.2718506
1: -15.4869547, 24.1380043, -25.2470398, 38.4499550, -53.9369087, 49.3850365
2: -15.3245106, 23.4516239, -25.0507812, 37.6142006, -52.9387054, 48.5024033
3: -19.7292652, 28.4352989, -31.7459335, 45.1771049, -64.9063721, 60.1812248
4: -17.6341152, 26.6464577, -28.4099579, 42.6639404, -60.2980576, 55.0564117

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033525, upper bound: 46.4000627
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -13.1799183, 23.5913467, -45.1264000, 51.0744629
1: -24.6167240, 37.5179062, -15.1432333, 23.5861683, -48.2028923, 52.6611404
2: -24.4222260, 36.7149124, -14.9693508, 22.9072952, -47.3295212, 51.6842575
3: -30.9534435, 44.0686035, -19.2966652, 27.7679253, -58.7213593, 63.3652687
4: -27.6933746, 41.6349754, -17.2341537, 26.0484867, -53.7418594, 58.8691216

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3936554, upper bound: 46.3913772
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000627, upper bound: 46.4033525
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -13.2305202, 23.6820087, -46.9176559, 53.6764793
1: -26.5506458, 40.1011238, -15.2022705, 23.6791344, -50.2297821, 55.3033943
2: -26.2936344, 39.2237549, -15.0285778, 22.9954662, -49.2891006, 54.2523346
3: -33.3686562, 47.1178703, -19.3709469, 27.8785477, -61.2471962, 66.4888153
4: -29.7111588, 44.5652657, -17.3076649, 26.1518784, -55.8630371, 61.8729286

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3918314
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999862, upper bound: 46.4038067
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -22.0821896, 38.8009224, -60.2906761, 59.9205170
1: -24.6167240, 37.5179062, -25.2398605, 38.4360886, -63.0528107, 62.7577591
2: -24.4222260, 36.7149124, -25.0442638, 37.6035576, -61.9845772, 61.6979942
3: -30.9534435, 44.0686035, -31.7336426, 45.1561775, -76.1096191, 75.8022461
4: -27.6933746, 41.6349754, -28.4019680, 42.6518250, -70.3451843, 70.0273438

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4073140, upper bound: 46.4107520
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4073140, upper bound: 46.4107520
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -22.1220589, 38.8677750, -62.0688934, 62.5367813
1: -26.5506458, 40.1011238, -25.2854958, 38.5041084, -65.0547485, 65.3866119
2: -26.2936344, 39.2237549, -25.0905609, 37.6694794, -63.9377556, 64.2713013
3: -33.3686562, 47.1178703, -31.7907104, 45.2369728, -78.6056290, 78.9085693
4: -29.7111588, 44.5652657, -28.4550457, 42.7269135, -72.4380722, 73.0174561

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3624684, upper bound: 46.3248131
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4046520, upper bound: 46.4089909
time: 1.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.94 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3810150, upper bound: 46.3810150
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3810150, upper bound: 46.3912553
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3837861, upper bound: 46.3975769
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3913772, upper bound: 46.3936554
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3935789
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.4033525, upper bound: 46.4000627
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3936554, upper bound: 46.3913772
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.4000627, upper bound: 46.4033525
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3883948, upper bound: 46.3918314
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3999862, upper bound: 46.4038067
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.4073140, upper bound: 46.4107520
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.4073140, upper bound: 46.4107520
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.3624684, upper bound: 46.3248131
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 3, lower bound: -46.4046520, upper bound: 46.4089909

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -12.3053322, 22.3548565, -34.6601868, 34.6601868
1: -14.1560287, 22.3223190, -14.1560287, 22.3223190, -36.4783478, 36.4783478
2: -14.0357513, 21.6778736, -14.0357513, 21.6778736, -35.7136230, 35.7136230
3: -18.0700207, 26.2806225, -18.0700207, 26.2806225, -44.3506355, 44.3506317
4: -16.2465935, 24.5797844, -16.2465935, 24.5797844, -40.8263779, 40.8263779

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3807871, upper bound: 46.3762536
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3764196
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -13.4809980, 24.1804466, -36.4857788, 35.8358498
1: -14.1560287, 22.3223190, -15.4840717, 24.1342449, -38.2902756, 37.8063850
2: -14.0357513, 21.6778736, -15.3217497, 23.4482899, -37.4840393, 36.9996223
3: -18.0700207, 26.2806225, -19.7252674, 28.4306870, -46.5006981, 46.0058899
4: -16.2465935, 24.5797844, -17.6306801, 26.6422653, -42.8888588, 42.2104645

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3807871, upper bound: 46.3882288
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3883948
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3614626, 23.9731903, -12.6108932, 22.6005650, -35.9620285, 36.5840836
1: -15.3469019, 23.9236946, -14.4915876, 22.5902061, -37.9371071, 38.4152718
2: -15.1847820, 23.2452793, -14.3125496, 21.9445305, -37.1293106, 37.5578308
3: -19.5525150, 28.1806087, -18.4772301, 26.5859222, -46.1384315, 46.6578369
4: -17.4681835, 26.4113426, -16.4592762, 24.9483700, -42.4165497, 42.8706131

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.4114799, 24.0652580, -14.2026863, 25.0186310, -38.4301109, 38.2679291
1: -15.4055624, 24.0188236, -16.3036613, 25.0498886, -40.4554482, 40.3224831
2: -15.2436056, 23.3354874, -16.0687141, 24.3294525, -39.5730553, 39.4042015
3: -19.6272469, 28.2937508, -20.7672157, 29.5027351, -49.1299820, 49.0609665
4: -17.5434914, 26.5134220, -18.3772621, 27.7533875, -45.2968750, 44.8906631

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.1846371, 22.1428127, -21.4594727, 37.7859688, -49.9706039, 43.6022873
1: -14.0173941, 22.1073685, -24.5308170, 37.4080963, -51.4254913, 46.6381798
2: -13.8968678, 21.4703846, -24.3398361, 36.6077347, -50.5046005, 45.8102036
3: -17.8945942, 26.0253162, -30.8461475, 43.9393387, -61.8339272, 56.8714638
4: -16.0817909, 24.3422451, -27.6063862, 41.5083885, -57.5901794, 51.9486237

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3934129
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3935789
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.2436686, 22.2502861, -23.1599712, 40.3375244, -52.5811882, 45.4102554
1: -14.0863037, 22.2172012, -26.4646950, 39.9910202, -54.0773201, 48.6818962
2: -13.9660330, 21.5753860, -26.2102661, 39.1166573, -53.0826836, 47.7856522
3: -17.9826965, 26.1559525, -33.2618599, 46.9889030, -64.9716034, 59.4178123
4: -16.1673737, 24.4655857, -29.6240044, 44.4390984, -60.6064720, 54.0895920

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3934129
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3935789
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3614626, 23.9731903, -21.4112053, 37.6834641, -51.0449257, 45.3843956
1: -15.3469019, 23.9236946, -24.4758625, 37.3142738, -52.6611748, 48.3995552
2: -15.1847820, 23.2452793, -24.2812729, 36.5151901, -51.6999664, 47.5265465
3: -19.5525150, 28.1806087, -30.7802677, 43.8317909, -63.3842964, 58.9608688
4: -17.4681835, 26.4113426, -27.5333710, 41.4057770, -58.8739548, 53.9447021

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.4114799, 24.0652580, -23.1046104, 40.2229309, -53.6344109, 47.1698647
1: -15.4055624, 24.0188236, -26.4018021, 39.8872261, -55.2927856, 50.4206238
2: -15.2436056, 23.3354874, -26.1462250, 39.0129089, -54.2565117, 49.4817123
3: -19.6272469, 28.2937508, -33.1849937, 46.8695107, -66.4967499, 61.4787445
4: -17.5434914, 26.5134220, -29.5438652, 44.3231888, -61.8666725, 56.0572777

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -21.4594727, 37.7859688, -12.1846371, 22.1428127, -43.6022873, 49.9706039
1: -24.5308170, 37.4080963, -14.0173941, 22.1073685, -46.6381798, 51.4254913
2: -24.3398361, 36.6077347, -13.8968678, 21.4703846, -45.8101997, 50.5046005
3: -30.8461475, 43.9393387, -17.8945942, 26.0253162, -56.8714638, 61.8339310
4: -27.6063862, 41.5083885, -16.0817909, 24.3422451, -51.9486237, 57.5901794

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910759, upper bound: 46.3908494
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3852423, upper bound: 46.3817303
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.4112053, 37.6834641, -13.3614626, 23.9731903, -45.3843956, 51.0449257
1: -24.4758625, 37.3142738, -15.3469019, 23.9236946, -48.3995552, 52.6611748
2: -24.2812729, 36.5151901, -15.1847820, 23.2452793, -47.5265465, 51.6999664
3: -30.7802677, 43.8317909, -19.5525150, 28.1806087, -58.9608727, 63.3843040
4: -27.5333710, 41.4057770, -17.4681835, 26.4113426, -53.9447021, 58.8739548

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953593, upper bound: 46.3999399
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -23.1599712, 40.3375244, -12.2436686, 22.2502861, -45.4102554, 52.5811882
1: -26.4646950, 39.9910202, -14.0863037, 22.2172012, -48.6818962, 54.0773239
2: -26.2102661, 39.1166573, -13.9660330, 21.5753860, -47.7856522, 53.0826836
3: -33.2618599, 46.9889030, -17.9826965, 26.1559525, -59.4178123, 64.9716034
4: -29.6240044, 44.4390984, -16.1673737, 24.4655857, -54.0895920, 60.6064720

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3935789, upper bound: 46.3913036
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3851399, upper bound: 46.3913036
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -23.1046104, 40.2229309, -13.4110756, 24.0647354, -47.1693420, 53.6340065
1: -26.4018021, 39.8872261, -15.4051037, 24.0182266, -50.4200249, 55.2923279
2: -26.1462250, 39.0129089, -15.2431688, 23.3349609, -49.4811821, 54.2560730
3: -33.1849937, 46.8695107, -19.6266136, 28.2930183, -61.4780121, 66.4961243
4: -29.5438652, 44.3231888, -17.5429401, 26.5127525, -56.0566177, 61.8661270

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3909760, upper bound: 46.4005517
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -21.5350533, 37.8945465, -59.3681831, 59.3681831
1: -24.6167240, 37.5179062, -24.6167240, 37.5179062, -62.1346283, 62.1346283
2: -24.4222260, 36.7149124, -24.4222260, 36.7149124, -61.0790482, 61.0790482
3: -30.9534435, 44.0686035, -30.9534435, 44.0686035, -75.0220490, 75.0220490
4: -27.6933746, 41.6349754, -27.6933746, 41.6349754, -69.3153381, 69.3153381

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068564, upper bound: 46.4067874
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -23.2356472, 40.4459610, -61.9330559, 61.0659599
1: -24.6167240, 37.5179062, -26.5506458, 40.1011238, -64.7178497, 64.0685501
2: -24.4222260, 36.7149124, -26.2936344, 39.2237549, -63.5969543, 62.9535904
3: -30.9534435, 44.0686035, -33.3686562, 47.1178703, -78.0713120, 77.4372559
4: -27.6933746, 41.6349754, -29.7111588, 44.5652657, -72.2460861, 71.3358154

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068564, upper bound: 46.4081789
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4080213
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.1599712, 40.3375244, -21.0309467, 37.3079491, -60.3716736, 61.3144608
1: -26.4646950, 39.9910202, -24.0480404, 36.9268112, -63.3801422, 64.0390625
2: -26.2102661, 39.1166573, -23.9025860, 36.1310844, -62.2657776, 62.9463806
3: -33.2618599, 46.9889030, -30.2495327, 43.3805313, -76.6423874, 77.2384338
4: -29.6240044, 44.4390984, -27.2052097, 40.9117889, -70.4983215, 71.6053162

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3082903, upper bound: 46.3082903
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3082903, upper bound: 46.3248131
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.1046104, 40.2229309, -21.9111729, 38.5895424, -61.6573830, 62.1220360
1: -26.4018021, 39.8872261, -25.0453281, 38.2216568, -64.6234589, 64.9325562
2: -26.1462250, 39.0129089, -24.8728600, 37.3748055, -63.4843979, 63.8563423
3: -33.1849937, 46.8695107, -31.5069122, 44.9283981, -78.1133881, 78.3764038
4: -29.5438652, 44.3231888, -28.2392769, 42.3796730, -71.9179077, 72.5624695

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4042531, upper bound: 46.4051431
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4025940, upper bound: 46.4051431
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3807871, upper bound: 46.3762536
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3764196
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3807871, upper bound: 46.3882288
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3764196, upper bound: 46.3883948
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3948020, upper bound: 46.3948021
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3934129
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3935789
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3934129
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3918314, upper bound: 46.3935789
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4033525, upper bound: 46.3999862
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4038066, upper bound: 46.3999862
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3910759, upper bound: 46.3908494
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3852423, upper bound: 46.3817303
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3953593, upper bound: 46.3999399
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3935789, upper bound: 46.3913036
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3851399, upper bound: 46.3913036
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3909760, upper bound: 46.4005517
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4068564, upper bound: 46.4067874
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4068564, upper bound: 46.4081789
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4080213
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3082903, upper bound: 46.3082903
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3082903, upper bound: 46.3248131
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4042531, upper bound: 46.4051431
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.4025940, upper bound: 46.4051431

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -12.1846371, 22.1428127, -33.8559189, 33.5022354
1: -13.4768963, 21.2664337, -14.0173941, 22.1073685, -35.5842667, 35.2838287
2: -13.3540964, 20.6594162, -13.8968678, 21.4703846, -34.8244781, 34.5562820
3: -17.2112961, 25.0253677, -17.8945942, 26.0253162, -43.2366104, 42.9199600
4: -15.4359884, 23.4152107, -16.0817909, 24.3422451, -39.7782288, 39.4970016

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -12.2436686, 22.2502861, -35.5475616, 35.9614296
1: -15.2800694, 23.7082329, -14.0863037, 22.2172012, -37.4972687, 37.7945328
2: -15.0858688, 23.0311298, -13.9660330, 21.5753860, -36.6612549, 36.9971619
3: -19.4841919, 27.9129372, -17.9826965, 26.1559525, -45.6401443, 45.8956184
4: -17.3320103, 26.2054291, -16.1673737, 24.4655857, -41.7975960, 42.3728027

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -13.3597670, 23.9710007, -35.6841087, 34.6773643
1: -13.4768963, 21.2664337, -15.3449726, 23.9211884, -37.3980865, 36.6114044
2: -13.3540964, 20.6594162, -15.1829395, 23.2430687, -36.5971603, 35.8423500
3: -17.2112961, 25.0253677, -19.5498638, 28.1775436, -45.3888359, 44.5752296
4: -15.4359884, 23.4152107, -17.4659004, 26.4085541, -41.8445435, 40.8811111

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -13.4110756, 24.0647354, -37.3620110, 37.1288338
1: -15.2800694, 23.7082329, -15.4051037, 24.0182266, -39.2982941, 39.1133347
2: -15.0858688, 23.0311298, -15.2431688, 23.3349609, -38.4208183, 38.2742958
3: -19.4841919, 27.9129372, -19.6266136, 28.2930183, -47.7772102, 47.5395508
4: -17.3320103, 26.2054291, -17.5429401, 26.5127525, -43.8447647, 43.7483673

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -12.6108932, 22.6005650, -35.4464722, 35.6982460
1: -14.7570267, 23.0229206, -14.4915876, 22.5902061, -37.3472328, 37.5145073
2: -14.5964413, 22.3779869, -14.3125496, 21.9445305, -36.5409698, 36.6905365
3: -18.8066769, 27.1103859, -18.4772301, 26.5859222, -45.3925934, 45.5876122
4: -16.7738895, 25.4232101, -16.4592762, 24.9483700, -41.7222595, 41.8824844

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3873140
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3971463
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -12.6108932, 22.6005650, -37.0344429, 38.1042328
1: -16.5651245, 25.4739952, -14.4915876, 22.5902061, -39.1553307, 39.9655800
2: -16.3321190, 24.7531395, -14.3125496, 21.9445305, -38.2766457, 39.0656776
3: -21.0962143, 30.0184078, -18.4772301, 26.5859222, -47.6821327, 48.4956322
4: -18.6865940, 28.2100029, -16.4592762, 24.9483700, -43.6349640, 44.6692810

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3873140
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3971463
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -14.2026863, 25.0186310, -37.8645401, 37.2900238
1: -14.7570267, 23.0229206, -16.3036613, 25.0498886, -39.8069153, 39.3265800
2: -14.5964413, 22.3779869, -16.0687141, 24.3294525, -38.9258919, 38.4467010
3: -18.8066769, 27.1103859, -20.7672157, 29.5027351, -48.3094101, 47.8776016
4: -16.7738895, 25.4232101, -18.3772621, 27.7533875, -44.5272751, 43.8004646

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3948021
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -14.2026863, 25.0186310, -39.4525108, 39.6960144
1: -16.5651245, 25.4739952, -16.3036613, 25.0498886, -41.6150131, 41.7776527
2: -16.3321190, 24.7531395, -16.0687141, 24.3294525, -40.6615601, 40.8218384
3: -21.0962143, 30.0184078, -20.7672157, 29.5027351, -50.5989494, 50.7856178
4: -18.6865940, 28.2100029, -18.3772621, 27.7533875, -46.4399796, 46.5872612

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3948021
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -21.4594727, 37.7859688, -49.4990768, 42.7770691
1: -13.4768963, 21.2664337, -24.5308170, 37.4080963, -50.8849945, 45.7972488
2: -13.3540964, 20.6594162, -24.3398361, 36.6077347, -49.9618301, 44.9992371
3: -17.2112961, 25.0253677, -30.8461475, 43.9393387, -61.1506310, 55.8715134
4: -15.4359884, 23.4152107, -27.6063862, 41.5083885, -56.9443779, 51.0215912

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3933564
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3892242
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -21.4594727, 37.7859688, -51.0832443, 45.1772346
1: -15.2800694, 23.7082329, -24.5308170, 37.4080963, -52.6881638, 48.2390480
2: -15.0858688, 23.0311298, -24.3398361, 36.6077347, -51.6935997, 47.3709564
3: -19.4841919, 27.9129372, -30.8461475, 43.9393387, -63.4235268, 58.7590790
4: -17.3320103, 26.2054291, -27.6063862, 41.5083885, -58.8403969, 53.8118057

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3936554
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895232
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -23.1599712, 40.3375244, -52.0506248, 44.4775696
1: -13.4768963, 21.2664337, -26.4646950, 39.9910202, -53.4679184, 47.7311287
2: -13.3540964, 20.6594162, -26.2102661, 39.1166573, -52.4707451, 46.8696823
3: -17.2112961, 25.0253677, -33.2618599, 46.9889030, -64.2001953, 58.2872276
4: -15.4359884, 23.4152107, -29.6240044, 44.4390984, -59.8750877, 53.0392151

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3932799
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894322
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -23.1599712, 40.3375244, -53.6348000, 46.8777313
1: -15.2800694, 23.7082329, -26.4646950, 39.9910202, -55.2710876, 50.1729240
2: -15.0858688, 23.0311298, -26.2102661, 39.1166573, -54.2025185, 49.2413940
3: -19.4841919, 27.9129372, -33.2618599, 46.9889030, -66.4730988, 61.1747971
4: -17.3320103, 26.2054291, -29.6240044, 44.4390984, -61.7711105, 55.8294334

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3935789
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -21.4112053, 37.6834641, -50.5293732, 44.4985580
1: -14.7570267, 23.0229206, -24.4758625, 37.3142738, -52.0713005, 47.4987831
2: -14.5964413, 22.3779869, -24.2812729, 36.5151901, -51.1116257, 46.6592560
3: -18.8066769, 27.1103859, -30.7802677, 43.8317909, -62.6384621, 57.8906441
4: -16.7738895, 25.4232101, -27.5333710, 41.4057770, -58.1796646, 52.9565735

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -21.4112053, 37.6834641, -52.1173401, 46.9045486
1: -16.5651245, 25.4739952, -24.4758625, 37.3142738, -53.8793983, 49.9498596
2: -16.3321190, 24.7531395, -24.2812729, 36.5151901, -52.8473053, 49.0344009
3: -21.0962143, 30.0184078, -30.7802677, 43.8317909, -64.9279938, 60.7986679
4: -18.6865940, 28.2100029, -27.5333710, 41.4057770, -60.0923691, 55.7433701

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -23.1046104, 40.2229309, -53.0688400, 46.1919556
1: -14.7570267, 23.0229206, -26.4018021, 39.8872261, -54.6442528, 49.4247208
2: -14.5964413, 22.3779869, -26.1462250, 39.0129089, -53.6093483, 48.5242119
3: -18.8066769, 27.1103859, -33.1849937, 46.8695107, -65.6761856, 60.2953796
4: -16.7738895, 25.4232101, -29.5438652, 44.3231888, -61.0970764, 54.9670753

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000976, upper bound: 46.3994150
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3955673
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -23.1046104, 40.2229309, -54.6568108, 48.5979500
1: -16.5651245, 25.4739952, -26.4018021, 39.8872261, -56.4523468, 51.8757935
2: -16.3321190, 24.7531395, -26.1462250, 39.0129089, -55.3450203, 50.8993607
3: -21.0962143, 30.0184078, -33.1849937, 46.8695107, -67.9657288, 63.2033920
4: -18.6865940, 28.2100029, -29.5438652, 44.3231888, -63.0097809, 57.7538681

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000976, upper bound: 46.3994150
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3909759, upper bound: 46.3954985
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.4062691, 37.7084198, -12.1150904, 22.0299225, -43.4361877, 49.8235092
1: -24.4727249, 37.4118919, -13.9388971, 21.9996738, -46.4723930, 51.3507881
2: -24.3275986, 36.5847321, -13.8191824, 21.3640289, -45.6916161, 50.4039154
3: -30.7865028, 43.9664688, -17.7971802, 25.8990135, -56.6855125, 61.7636490
4: -27.6669178, 41.5065880, -15.9981546, 24.2204323, -51.8873520, 57.5047417

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3933564, upper bound: 46.3908494
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3933564, upper bound: 46.3908494
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.1897602, 37.3581772, -12.1846371, 22.1428127, -43.3325729, 49.5428162
1: -24.2224827, 36.9804840, -14.0173941, 22.1073685, -46.3298492, 50.9978790
2: -24.0437469, 36.1881027, -13.8968678, 21.4703846, -45.5141258, 50.0849686
3: -30.4590530, 43.4385376, -17.8945942, 26.0253162, -56.4843674, 61.3331299
4: -27.2850800, 41.0336189, -16.0817909, 24.3422451, -51.6273155, 57.1154099

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892242, upper bound: 46.3906917
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892242, upper bound: 46.3906917
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.3605766, 37.6100082, -13.2875376, 23.8552322, -45.2158089, 50.8975449
1: -24.4206333, 37.3216324, -15.2633009, 23.8103352, -48.2309685, 52.5849266
2: -24.2719193, 36.4956360, -15.1024837, 23.1337242, -47.4056435, 51.5981178
3: -30.7238159, 43.8625679, -19.4487648, 28.0475807, -58.7713928, 63.3113213
4: -27.5970783, 41.4069023, -17.3795929, 26.2821712, -53.8792381, 58.7864914

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.1425095, 37.2571831, -13.3614626, 23.9731903, -45.1156998, 50.6186447
1: -24.1686878, 36.8880615, -15.3469019, 23.9236946, -48.0923767, 52.2349625
2: -23.9862709, 36.0970192, -15.1847820, 23.2452793, -47.2315521, 51.2817993
3: -30.3945980, 43.3325729, -19.5525150, 28.1806087, -58.5752068, 62.8850861
4: -27.2131939, 40.9325562, -17.4681835, 26.4113426, -53.6245232, 58.4007339

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3999399
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953593, upper bound: 46.3999399
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.1141586, 40.2621117, -12.1738548, 22.1370506, -45.2512093, 52.4359665
1: -26.4162636, 39.9891129, -14.0075016, 22.1091576, -48.5254173, 53.9966011
2: -26.1943092, 39.0904465, -13.8880663, 21.4686604, -47.6629715, 52.9785080
3: -33.2090988, 47.0103989, -17.8849087, 26.0292454, -59.2383423, 64.8953094
4: -29.6880627, 44.4297180, -16.0834923, 24.3433418, -54.0313988, 60.5132103

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932799, upper bound: 46.3913036
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880959, upper bound: 46.3913036
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.8807831, 39.8991852, -12.2436686, 22.2502861, -45.1310577, 52.1428528
1: -26.1454449, 39.5512428, -14.0863037, 22.2172012, -48.3626480, 53.6375465
2: -25.9011059, 38.6854095, -13.9660330, 21.5753860, -47.4764938, 52.6514359
3: -32.8627701, 46.4761505, -17.9826965, 26.1559525, -59.0187225, 64.4588394
4: -29.2928905, 43.9514465, -16.1673737, 24.4655857, -53.7584763, 60.1188202

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3894322, upper bound: 46.3913036
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3894322, upper bound: 46.3913036
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.0601902, 40.1506310, -13.3379831, 23.9478683, -47.0080566, 53.4886093
1: -26.3547325, 39.8874016, -15.3224449, 23.9061203, -50.2608452, 55.2098389
2: -26.1292992, 38.9894867, -15.1617918, 23.2245045, -49.3537979, 54.1512718
3: -33.1345825, 46.8936310, -19.5241394, 28.1615429, -61.2961273, 66.4177704
4: -29.6081562, 44.3169746, -17.4555550, 26.3849716, -55.9931221, 61.7725143

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005518
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.8267231, 39.7864799, -13.4110756, 24.0647354, -46.8914490, 53.1975517
1: -26.0840530, 39.4488525, -15.4051037, 24.0182266, -50.1022797, 54.8539543
2: -25.8383369, 38.5836258, -15.2431688, 23.3349609, -49.1732903, 53.8267899
3: -32.7878799, 46.3585701, -19.6266136, 28.2930183, -61.0808983, 65.9851837
4: -29.2136402, 43.8378944, -17.5429401, 26.5127525, -55.7263947, 61.3808365

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005517
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005517
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -21.4569397, 37.7666130, -59.1978149, 59.2036781
1: -24.5586433, 37.5217667, -24.5282040, 37.3956680, -61.9543114, 62.0499649
2: -24.4099751, 36.6918221, -24.3352013, 36.5940628, -60.9345436, 60.9625473
3: -30.8934937, 44.0957451, -30.8440266, 43.9258461, -74.8193359, 74.9397736
4: -27.7538719, 41.6327591, -27.6001701, 41.4974060, -69.2457123, 69.2279968

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -21.5350533, 37.8945465, -59.0963364, 58.9460907
1: -24.3087234, 37.0905914, -24.6167240, 37.5179062, -61.8266296, 61.7073135
2: -24.1263924, 36.2956047, -24.4222260, 36.7149124, -60.7806091, 60.6627350
3: -30.5667114, 43.5680695, -30.9534435, 44.0686035, -74.6353149, 74.5215149
4: -27.3722706, 41.1605835, -27.6933746, 41.6349754, -68.9950638, 68.8385544

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -23.1588268, 40.3195343, -61.7645569, 60.9027443
1: -24.5586433, 37.5217667, -26.4636326, 39.9801941, -64.5388336, 63.9853935
2: -24.4099751, 36.6918221, -26.2082100, 39.1044273, -63.4539795, 62.8386650
3: -30.8934937, 44.0957451, -33.2606850, 46.9767380, -77.8702316, 77.3564301
4: -27.7538719, 41.6327591, -29.6193867, 44.4293976, -72.1782990, 71.2498779

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4080213
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -23.2356472, 40.4459610, -61.6612091, 60.6438713
1: -24.3087234, 37.0905914, -26.5506458, 40.1011238, -64.4098434, 63.6412354
2: -24.1263924, 36.2956047, -26.2936344, 39.2237549, -63.2985153, 62.5372772
3: -30.5667114, 43.5680695, -33.3686562, 47.1178703, -77.6845779, 76.9367218
4: -27.3722706, 41.1605835, -29.7111588, 44.5652657, -71.9258041, 70.8590393

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.9733620, 40.0979080, -21.0309467, 37.3079491, -60.2004433, 61.0750771
1: -26.2538033, 39.7481461, -24.0480404, 36.9268112, -63.1806145, 63.7961884
2: -26.0135250, 38.8587685, -23.9025860, 36.1310844, -62.0860901, 62.6802254
3: -33.0165367, 46.7308502, -30.2495327, 43.3805313, -76.3970642, 76.9803848
4: -29.4408531, 44.1387405, -27.2052097, 40.9117889, -70.3241043, 71.2858047

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2988632, upper bound: 46.2865483
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3071427, upper bound: 46.3237231
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.0601902, 40.1506310, -21.8303661, 38.4576874, -61.4931755, 61.9613838
1: -26.3547325, 39.8874016, -24.9536953, 38.0952530, -64.4499817, 64.8410950
2: -26.1292992, 38.9894867, -24.7830334, 37.2501373, -63.3455544, 63.7384682
3: -33.1345825, 46.8936310, -31.3939590, 44.7808456, -77.9154282, 78.2875748
4: -29.6081562, 44.3169746, -28.1430550, 42.2378273, -71.8459778, 72.4600296

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4041220, upper bound: 46.4049350
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4041220, upper bound: 46.4051431
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.8267231, 39.7864799, -21.9111729, 38.5895424, -61.3770409, 61.6907921
1: -26.0840530, 39.4488525, -25.0453281, 38.2216568, -64.3057022, 64.4941788
2: -25.8383369, 38.5836258, -24.8728600, 37.3748055, -63.1762619, 63.4304543
3: -32.7878799, 46.3585701, -31.5069122, 44.9283981, -77.7162781, 77.8654709
4: -29.2136402, 43.8378944, -28.2392769, 42.3796730, -71.5886459, 72.0771637

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4020868, upper bound: 46.4049351
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4020868, upper bound: 46.4051430
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.42 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3762536
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3762536, upper bound: 46.3764196
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3882288
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3828268, upper bound: 46.3883948
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3873140
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3971463
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3873140
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3971463
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3948021
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3828268
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3882288, upper bound: 46.3948021
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3933564
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3892242
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3936554
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895232
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3932799
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894322
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3908494, upper bound: 46.3935789
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4000975, upper bound: 46.3994915
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4000976, upper bound: 46.3994150
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3999399, upper bound: 46.3955673
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4000976, upper bound: 46.3994150
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3909759, upper bound: 46.3954985
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3933564, upper bound: 46.3908494
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3933564, upper bound: 46.3908494
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3892242, upper bound: 46.3906917
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3892242, upper bound: 46.3906917
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3994915, upper bound: 46.4000975
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3999399
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3953593, upper bound: 46.3999399
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3932799, upper bound: 46.3913036
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3880959, upper bound: 46.3913036
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3894322, upper bound: 46.3913036
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3894322, upper bound: 46.3913036
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005517
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4005518
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005517
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3955673, upper bound: 46.4005517
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4028899, upper bound: 46.4066297
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3953800, upper bound: 46.4080213
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4031142, upper bound: 46.4080213
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.2988632, upper bound: 46.2865483
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.3071427, upper bound: 46.3237231
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4041220, upper bound: 46.4049350
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4041220, upper bound: 46.4051431
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4020868, upper bound: 46.4049351
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -46.4020868, upper bound: 46.4051430

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -11.7131090, 21.3175964, -33.0307045, 33.0307045
1: -13.4768963, 21.2664337, -13.4768963, 21.2664337, -34.7433319, 34.7433319
2: -13.3540964, 20.6594162, -13.3540964, 20.6594162, -34.0135117, 34.0135117
3: -17.2112961, 25.0253677, -17.2112961, 25.0253677, -42.2366638, 42.2366638
4: -15.4359884, 23.4152107, -15.4359884, 23.4152107, -38.8512001, 38.8512001

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806078, upper bound: 46.3757257
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3755927
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -13.2972765, 23.7177620, -35.4308701, 34.6148720
1: -13.4768963, 21.2664337, -15.2800694, 23.7082329, -37.1851273, 36.5465012
2: -13.3540964, 20.6594162, -15.0858688, 23.0311298, -36.3852272, 35.7452850
3: -17.2112961, 25.0253677, -19.4841919, 27.9129372, -45.1242294, 44.5095596
4: -15.4359884, 23.4152107, -17.3320103, 26.2054291, -41.6414146, 40.7472229

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806078, upper bound: 46.3757257
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3755927
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -11.7131090, 21.3175964, -34.6148720, 35.4308701
1: -15.2800694, 23.7082329, -13.4768963, 21.2664337, -36.5465012, 37.1851273
2: -15.0858688, 23.0311298, -13.3540964, 20.6594162, -35.7452850, 36.3852272
3: -19.4841919, 27.9129372, -17.2112961, 25.0253677, -44.5095596, 45.1242294
4: -17.3320103, 26.2054291, -15.4359884, 23.4152107, -40.7472229, 41.6414146

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3761206, upper bound: 46.3757726
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3758917
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -13.2972765, 23.7177620, -37.0150375, 37.0150375
1: -15.2800694, 23.7082329, -15.2800694, 23.7082329, -38.9883041, 38.9883003
2: -15.0858688, 23.0311298, -15.0858688, 23.0311298, -38.1169968, 38.1169968
3: -19.4841919, 27.9129372, -19.4841919, 27.9129372, -47.3971291, 47.3971291
4: -17.3320103, 26.2054291, -17.3320103, 26.2054291, -43.5374336, 43.5374374

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3761206, upper bound: 46.3757726
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3755927, upper bound: 46.3758917
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -12.8459072, 23.0873528, -34.8004608, 34.1635056
1: -13.4768963, 21.2664337, -14.7570267, 23.0229206, -36.4998169, 36.0234604
2: -13.3540964, 20.6594162, -14.5964413, 22.3779869, -35.7320824, 35.2558594
3: -17.2112961, 25.0253677, -18.8066769, 27.1103859, -44.3216782, 43.8320389
4: -15.4359884, 23.4152107, -16.7738895, 25.4232101, -40.8591995, 40.1891022

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3867428, upper bound: 46.3849739
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3848409
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -14.4310522, 25.4894829, -37.2025833, 35.7486496
1: -13.4768963, 21.2664337, -16.5618916, 25.4696083, -38.9465027, 37.8283234
2: -13.3540964, 20.6594162, -16.3289795, 24.7492275, -38.1033249, 36.9883881
3: -17.2112961, 25.0253677, -21.0917263, 30.0130558, -47.2243500, 46.1170845
4: -15.4359884, 23.4152107, -18.6827087, 28.2050896, -43.6410751, 42.0979195

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3867428, upper bound: 46.3849739
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3848409
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -12.8459072, 23.0873528, -36.3846283, 36.5636673
1: -15.2800694, 23.7082329, -14.7570267, 23.0229206, -38.3029900, 38.4652596
2: -15.0858688, 23.0311298, -14.5964413, 22.3779869, -37.4638557, 37.6275711
3: -19.4841919, 27.9129372, -18.8066769, 27.1103859, -46.5945778, 46.7196007
4: -17.3320103, 26.2054291, -16.7738895, 25.4232101, -42.7552185, 42.9793167

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3822556, upper bound: 46.3850207
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3851399
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -14.4310522, 25.4894829, -38.7867584, 38.1488113
1: -15.2800694, 23.7082329, -16.5618916, 25.4696083, -40.7496796, 40.2701263
2: -15.0858688, 23.0311298, -16.3289795, 24.7492275, -39.8350983, 39.3601074
3: -19.4841919, 27.9129372, -21.0917263, 30.0130558, -49.4972458, 49.0046539
4: -17.3320103, 26.2054291, -18.6827087, 28.2050896, -45.5370979, 44.8881378

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3822556, upper bound: 46.3850208
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3851399
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -11.7131090, 21.3175964, -34.1635056, 34.8004570
1: -14.7570267, 23.0229206, -13.4768963, 21.2664337, -36.0234604, 36.4998169
2: -14.5964413, 22.3779869, -13.3540964, 20.6594162, -35.2558594, 35.7320824
3: -18.8066769, 27.1103859, -17.2112961, 25.0253677, -43.8320351, 44.3216820
4: -16.7738895, 25.4232101, -15.4359884, 23.4152107, -40.1891022, 40.8591995

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693105, upper bound: 46.3754472
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892987, upper bound: 46.3857845
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -12.8459072, 23.0873528, -35.9332581, 35.9332581
1: -14.7570267, 23.0229206, -14.7570267, 23.0229206, -37.7799454, 37.7799454
2: -14.5964413, 22.3779869, -14.5964413, 22.3779869, -36.9744263, 36.9744263
3: -18.8066769, 27.1103859, -18.8066769, 27.1103859, -45.9170532, 45.9170532
4: -16.7738895, 25.4232101, -16.7738895, 25.4232101, -42.1970978, 42.1970978

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693105, upper bound: 46.3789049
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3892988, upper bound: 46.3971113
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -11.7131090, 21.3175964, -35.7514763, 37.2064476
1: -16.5651245, 25.4739952, -13.4768963, 21.2664337, -37.8315582, 38.9508896
2: -16.3321190, 24.7531395, -13.3540964, 20.6594162, -36.9915237, 38.1072235
3: -21.0962143, 30.0184078, -17.2112961, 25.0253677, -46.1215782, 47.2296944
4: -18.6865940, 28.2100029, -15.4359884, 23.4152107, -42.1018066, 43.6459923

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880958, upper bound: 46.3822990
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3817278
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -12.8459072, 23.0873528, -37.5212288, 38.3392487
1: -16.5651245, 25.4739952, -14.7570267, 23.0229206, -39.5880432, 40.2310219
2: -16.3321190, 24.7531395, -14.5964413, 22.3779869, -38.7101021, 39.3495712
3: -21.0962143, 30.0184078, -18.8066769, 27.1103859, -48.2065964, 48.8250732
4: -18.6865940, 28.2100029, -16.7738895, 25.4232101, -44.1098022, 44.9838943

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880959, upper bound: 46.3916495
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3910784
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -13.3249502, 23.7547188, -36.6006241, 36.4123039
1: -14.7570267, 23.0229206, -15.3124533, 23.7481346, -38.5051613, 38.3353729
2: -14.5964413, 22.3779869, -15.1157141, 23.0690403, -37.6654816, 37.4937019
3: -18.8066769, 27.1103859, -19.5282536, 27.9609547, -46.7676239, 46.6386414
4: -16.7738895, 25.4232101, -17.3669949, 26.2523479, -43.0262375, 42.7902069

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693069, upper bound: 46.3709700
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693069, upper bound: 46.3812211
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -14.4338799, 25.4933434, -38.3392487, 37.5212288
1: -14.7570267, 23.0229206, -16.5651245, 25.4739952, -40.2310219, 39.5880432
2: -14.5964413, 22.3779869, -16.3321190, 24.7531395, -39.3495712, 38.7101021
3: -18.8066769, 27.1103859, -21.0962143, 30.0184078, -48.8250732, 48.2065926
4: -16.7738895, 25.4232101, -18.6865940, 28.2100029, -44.9838943, 44.1098022

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693069, upper bound: 46.3759340
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3893952, upper bound: 46.3941068
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -13.3249502, 23.7547188, -38.1885910, 38.8182945
1: -16.5651245, 25.4739952, -15.3124533, 23.7481346, -40.3132591, 40.7864494
2: -16.3321190, 24.7531395, -15.1157141, 23.0690403, -39.4011574, 39.8688393
3: -21.0962143, 30.0184078, -19.5282536, 27.9609547, -49.0571671, 49.5466614
4: -18.6865940, 28.2100029, -17.3669949, 26.2523479, -44.9389420, 45.5769958

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880958, upper bound: 46.3822990
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3817278
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -14.4338799, 25.4933434, -39.9272194, 39.9272194
1: -16.5651245, 25.4739952, -16.5651245, 25.4739952, -42.0391197, 42.0391197
2: -16.3321190, 24.7531395, -16.3321190, 24.7531395, -41.0852432, 41.0852394
3: -21.0962143, 30.0184078, -21.0962143, 30.0184078, -51.1146164, 51.1146164
4: -18.6865940, 28.2100029, -18.6865940, 28.2100029, -46.8965988, 46.8965988

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880959, upper bound: 46.3915472
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848409, upper bound: 46.3909760
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.6497908, 21.2148361, -21.4062691, 37.7084198, -49.3582115, 42.6210976
1: -13.4053020, 21.1696339, -24.4727249, 37.4118919, -50.8171921, 45.6423531
2: -13.2836123, 20.5638943, -24.3275986, 36.5847321, -49.8683434, 44.8914795
3: -17.1225319, 24.9118900, -30.7865028, 43.9664688, -61.0890007, 55.6983948
4: -15.3605337, 23.3052635, -27.6669178, 41.5065880, -56.8671150, 50.9721756

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3700504, upper bound: 46.3868096
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3890542, upper bound: 46.3933564
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -21.1897602, 37.3581772, -49.0712852, 42.5073547
1: -13.4768963, 21.2664337, -24.2224827, 36.9804840, -50.4573822, 45.4889145
2: -13.3540964, 20.6594162, -24.0437469, 36.1881027, -49.5421982, 44.7031631
3: -17.2112961, 25.0253677, -30.4590530, 43.4385376, -60.6498337, 55.4844208
4: -15.4359884, 23.4152107, -27.2850800, 41.0336189, -56.4696083, 50.7002907

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3849433
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3892242
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2285137, 23.6064682, -21.4062691, 37.7084198, -50.9369316, 45.0127296
1: -15.2023802, 23.6022644, -24.4727249, 37.4118919, -52.6142731, 48.0749855
2: -15.0094299, 22.9263000, -24.3275986, 36.5847321, -51.5941620, 47.2538872
3: -19.3880405, 27.7887878, -30.7865028, 43.9664688, -63.3545074, 58.5752907
4: -17.2496662, 26.0852890, -27.6669178, 41.5065880, -58.7562561, 53.7522049

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894040
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895232
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -21.1897602, 37.3581772, -50.6554527, 44.9075241
1: -15.2800694, 23.7082329, -24.2224827, 36.9804840, -52.2605515, 47.9307175
2: -15.0858688, 23.0311298, -24.0437469, 36.1881027, -51.2739716, 47.0748749
3: -19.4841919, 27.9129372, -30.4590530, 43.4385376, -62.9227295, 58.3719864
4: -17.3320103, 26.2054291, -27.2850800, 41.0336189, -58.3656273, 53.4905052

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3894040
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817303, upper bound: 46.3895232
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.6497908, 21.2148361, -23.1141586, 40.2621117, -51.9119034, 44.3289871
1: -13.4053020, 21.1696339, -26.4162636, 39.9891129, -53.3944054, 47.5858917
2: -13.2836123, 20.5638943, -26.1943092, 39.0904465, -52.3740501, 46.7582016
3: -17.1225319, 24.9118900, -33.2090988, 47.0103989, -64.1329346, 58.1209831
4: -15.3605337, 23.3052635, -29.6880627, 44.4297180, -59.7902451, 52.9933167

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3690810, upper bound: 46.3836810
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913036, upper bound: 46.3932799
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -22.8807831, 39.8991852, -51.6122894, 44.1983795
1: -13.4768963, 21.2664337, -26.1454449, 39.5512428, -53.0281372, 47.4118805
2: -13.3540964, 20.6594162, -25.9011059, 38.6854095, -52.0394974, 46.5605164
3: -17.2112961, 25.0253677, -32.8627701, 46.4761505, -63.6874466, 57.8881340
4: -15.4359884, 23.4152107, -29.2928905, 43.9514465, -59.3874321, 52.7080994

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3913036, upper bound: 46.3894322
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3848409
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2285137, 23.6064682, -23.1141586, 40.2621117, -53.4906235, 46.7206268
1: -15.2023802, 23.6022644, -26.4162636, 39.9891129, -55.1914940, 50.0185242
2: -15.0094299, 22.9263000, -26.1943092, 39.0904465, -54.0998688, 49.1206093
3: -19.3880405, 27.7887878, -33.2090988, 47.0103989, -66.3984375, 60.9978790
4: -17.2496662, 26.0852890, -29.6880627, 44.4297180, -61.6793823, 55.7733459

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3817278, upper bound: 46.3895200
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -22.8807831, 39.8991852, -53.1964607, 46.5985451
1: -15.2800694, 23.7082329, -26.1454449, 39.5512428, -54.8313141, 49.8536758
2: -15.0858688, 23.0311298, -25.9011059, 38.6854095, -53.7712746, 48.9322357
3: -19.4841919, 27.9129372, -32.8627701, 46.4761505, -65.9603424, 60.7756996
4: -17.3320103, 26.2054291, -29.2928905, 43.9514465, -61.2834549, 55.4983177

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3895200
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3906917, upper bound: 46.3897312
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.7749825, 22.9735832, -21.3605766, 37.6100082, -50.3849831, 44.3341599
1: -14.6768446, 22.9139671, -24.4206333, 37.3216324, -51.9984741, 47.3346024
2: -14.5173721, 22.2705345, -24.2719193, 36.4956360, -51.0130081, 46.5424538
3: -18.7072830, 26.9825783, -30.7238159, 43.8625679, -62.5698509, 57.7063866
4: -16.6888599, 25.2989311, -27.5970783, 41.4069023, -58.0957642, 52.8960037

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3830999, upper bound: 46.3895497
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986878, upper bound: 46.3986448
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -21.1425095, 37.2571831, -50.1030884, 44.2298622
1: -14.7570267, 23.0229206, -24.1686878, 36.8880615, -51.6450882, 47.1916084
2: -14.5964413, 22.3779869, -23.9862709, 36.0970192, -50.6934586, 46.3642578
3: -18.8066769, 27.1103859, -30.3945980, 43.3325729, -62.1392479, 57.5049820
4: -16.7738895, 25.4232101, -27.2131939, 40.9325562, -57.7064438, 52.6363945

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828120, upper bound: 46.3853589
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985102, upper bound: 46.3944209
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3628931, 25.3797798, -21.3605766, 37.6100082, -51.9728966, 46.7403526
1: -16.4847546, 25.3652096, -24.4206333, 37.3216324, -53.8063889, 49.7858429
2: -16.2532310, 24.6457901, -24.2719193, 36.4956360, -52.7488670, 48.9177094
3: -20.9963589, 29.8907681, -30.7238159, 43.8625679, -64.8589172, 60.6145821
4: -18.6016941, 28.0860825, -27.5970783, 41.4069023, -60.0085983, 55.6831589

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3805681, upper bound: 46.3891578
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985203, upper bound: 46.3986152
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -21.1425095, 37.2571831, -51.6910591, 46.6358528
1: -16.5651245, 25.4739952, -24.1686878, 36.8880615, -53.4531860, 49.6426849
2: -16.3321190, 24.7531395, -23.9862709, 36.0970192, -52.4291382, 48.7393951
3: -21.0962143, 30.0184078, -30.3945980, 43.3325729, -64.4287872, 60.4130058
4: -18.6865940, 28.2100029, -27.2131939, 40.9325562, -59.6191444, 55.4231911

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999400, upper bound: 46.3953593
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.7749825, 22.9735832, -23.0601902, 40.1506310, -52.9256096, 46.0337753
1: -14.6768446, 22.9139671, -26.3547325, 39.8874016, -54.5642433, 49.2686996
2: -14.5173721, 22.2705345, -26.1292992, 38.9894867, -53.5068550, 48.3998337
3: -18.7072830, 26.9825783, -33.1345825, 46.8936310, -65.6009064, 60.1171608
4: -16.6888599, 25.2989311, -29.6081562, 44.3169746, -61.0058365, 54.9070892

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3840651, upper bound: 46.3895497
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990836, upper bound: 46.3986448
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -22.8267231, 39.7864799, -52.6323853, 45.9140663
1: -14.7570267, 23.0229206, -26.0840530, 39.4488525, -54.2058792, 49.1069717
2: -14.5964413, 22.3779869, -25.8383369, 38.5836258, -53.1800613, 48.2163239
3: -18.8066769, 27.1103859, -32.7878799, 46.3585701, -65.1652451, 59.8982658
4: -16.7738895, 25.4232101, -29.2136402, 43.8378944, -60.6117859, 54.6368484

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3840904, upper bound: 46.3862368
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990836, upper bound: 46.3947648
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3628931, 25.3797798, -23.0601902, 40.1506310, -54.5135231, 48.4399643
1: -16.4847546, 25.3652096, -26.3547325, 39.8874016, -56.3721542, 51.7199402
2: -16.2532310, 24.6457901, -26.1292992, 38.9894867, -55.2427177, 50.7750854
3: -20.9963589, 29.8907681, -33.1345825, 46.8936310, -67.8899841, 63.0253525
4: -18.6016941, 28.0860825, -29.6081562, 44.3169746, -62.9186707, 57.6942368

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3805681, upper bound: 46.3891578
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985203, upper bound: 46.3986152
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -22.8267231, 39.7864799, -54.2203560, 48.3200607
1: -16.5651245, 25.4739952, -26.0840530, 39.4488525, -56.0139771, 51.5580482
2: -16.3321190, 24.7531395, -25.8383369, 38.5836258, -54.9157333, 50.5914688
3: -21.0962143, 30.0184078, -32.7878799, 46.3585701, -67.4547882, 62.8062820
4: -18.6865940, 28.2100029, -29.2136402, 43.8378944, -62.5244865, 57.4236450

Time for backsubstitution: 2.69 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257
time: 0.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -15.5593853, 27.1077271, -40.4099159, 39.3618546
1: -15.2836475, 23.8005772, -17.8337097, 27.1816788, -42.4653244, 41.6342773
2: -15.1096706, 23.1136723, -17.5771866, 26.3973064, -41.5069771, 40.6908569
3: -19.4732094, 28.0227852, -22.6461716, 31.9880333, -51.4612236, 50.6689568
4: -17.4002285, 26.2846336, -20.0349903, 30.2143135, -47.6145363, 46.3196259

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4007247, upper bound: 46.4007247
time: 0.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4007247, upper bound: 46.4024257
time: 0.91 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -15.6051130, 27.1780758, -49.3911247, 54.6228714
1: -25.3891716, 38.6557007, -17.8864689, 27.2607346, -52.6499062, 56.5421677
2: -25.1929913, 37.8162651, -17.6271324, 26.4738178, -51.6667976, 55.4433975
3: -31.9208298, 45.4162788, -22.7172165, 32.0807762, -64.0016022, 68.1334839
4: -28.5711136, 42.8956490, -20.0892334, 30.3068199, -58.8779335, 62.9848824

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4024257, upper bound: 46.4091342
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4024257, upper bound: 46.4091342
time: 1.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 3, lower bound: -46.4007247, upper bound: 46.4007247
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 3, lower bound: -46.4007247, upper bound: 46.4024257
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 3, lower bound: -46.4024257, upper bound: 46.4091342
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 3, lower bound: -46.4024257, upper bound: 46.4091342

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.3021946, 23.8024693, -37.1046600, 37.1046600
1: -15.2836475, 23.8005772, -15.2836475, 23.8005772, -39.0842171, 39.0842209
2: -15.1096706, 23.1136723, -15.1096706, 23.1136723, -38.2233429, 38.2233429
3: -19.4732094, 28.0227852, -19.4732094, 28.0227852, -47.4959869, 47.4959869
4: -17.4002285, 26.2846336, -17.4002285, 26.2846336, -43.6848602, 43.6848602

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3877874, upper bound: 46.3909410
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986058, upper bound: 46.3986059
time: 0.68 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -22.2130566, 39.0177612, -52.3199539, 46.0155220
1: -15.2836475, 23.8005772, -25.3891716, 38.6557007, -53.9393463, 49.1897392
2: -15.1096706, 23.1136723, -25.1929913, 37.8162651, -52.9259338, 48.3066559
3: -19.4732094, 28.0227852, -31.9208298, 45.4162788, -64.8894882, 59.9436111
4: -17.4002285, 26.2846336, -28.5711136, 42.8956490, -60.2958755, 54.8557472

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3877874, upper bound: 46.3931448
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3877874, upper bound: 46.4009719
time: 0.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -13.3021946, 23.8024693, -46.0155220, 52.3199539
1: -25.3891716, 38.6557007, -15.2836475, 23.8005772, -49.1897392, 53.9393463
2: -25.1929913, 37.8162651, -15.1096706, 23.1136723, -48.3066559, 52.9259338
3: -31.9208298, 45.4162788, -19.4732094, 28.0227852, -59.9436111, 64.8894806
4: -28.5711136, 42.8956490, -17.4002285, 26.2846336, -54.8557472, 60.2958755

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4016958, upper bound: 46.4073135
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3983519, upper bound: 46.3983519
time: 0.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -22.2130566, 39.0177612, -61.2015877, 61.2015839
1: -25.3891716, 38.6557007, -25.3891716, 38.6557007, -64.0448761, 64.0448761
2: -25.1929913, 37.8162651, -25.1929913, 37.8162651, -62.9735107, 62.9735184
3: -31.9208298, 45.4162788, -31.9208298, 45.4162788, -77.3370972, 77.3370972
4: -28.5711136, 42.8956490, -28.5711136, 42.8956490, -71.4667664, 71.4667664

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3998717, upper bound: 46.4105700
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3983519, upper bound: 46.3983519
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3877874, upper bound: 46.3909410
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3986058, upper bound: 46.3986059
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3877874, upper bound: 46.3931448
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3877874, upper bound: 46.4009719
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.4016958, upper bound: 46.4073135
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3983519, upper bound: 46.3983519
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3998717, upper bound: 46.4105700
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -46.3983519, upper bound: 46.3983519

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -12.8755350, 23.1897926, -35.4951248, 35.2303925
1: -14.1560287, 22.3223190, -14.8043928, 23.1745014, -37.3305283, 37.1267052
2: -14.0357513, 21.6778736, -14.6530132, 22.5055084, -36.5412598, 36.3308868
3: -18.0700207, 26.2806225, -18.8773499, 27.2818756, -45.3518829, 45.1579742
4: -16.2465935, 24.5797844, -16.9097538, 25.5623875, -41.8089752, 41.4895325

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3827630, upper bound: 46.3901055
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -13.1290922, 23.4807301, -36.9642601, 37.3128166
1: -15.4869547, 24.1380043, -15.0837631, 23.4972515, -38.9842072, 39.2217598
2: -15.3245106, 23.4516239, -14.9012632, 22.8120308, -38.1365318, 38.3528862
3: -19.7292652, 28.4352989, -19.2207775, 27.6679268, -47.3971901, 47.6560745
4: -17.6341152, 26.6464577, -17.1591320, 25.9417019, -43.5758133, 43.8055840

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3827630, upper bound: 46.3975543
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -21.7480507, 38.3521614, -50.6574936, 44.1028976
1: -14.1560287, 22.3223190, -24.8614616, 37.9827957, -52.1388245, 47.1837769
2: -14.0357513, 21.6778736, -24.6863194, 37.1600189, -51.1957703, 46.3641930
3: -18.0700207, 26.2806225, -31.2638092, 44.6242142, -62.6942253, 57.5444298
4: -16.2465935, 24.5797844, -28.0376873, 42.1213112, -58.3678970, 52.6174698

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3827630, upper bound: 46.3930749
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -21.9056091, 38.4943962, -51.9779282, 46.0893326
1: -15.4869547, 24.1380043, -25.0394192, 38.1503029, -53.6372566, 49.1774216
2: -15.3245106, 23.4516239, -24.8432961, 37.3204117, -52.6449127, 48.2949142
3: -19.7292652, 28.4352989, -31.4904728, 44.8285637, -64.5578308, 59.9257698
4: -17.6341152, 26.6464577, -28.1751442, 42.3258934, -59.9600067, 54.8215942

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3827630, upper bound: 46.4000627
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4035650, upper bound: 46.3999862
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -12.9811535, 23.2484474, -44.7835007, 50.8756943
1: -24.6167240, 37.5179062, -14.9156809, 23.2379169, -47.8546410, 52.4335861
2: -24.4222260, 36.7149124, -14.7413034, 22.5721760, -46.9943962, 51.4562149
3: -30.9534435, 44.0686035, -19.0106678, 27.3538895, -58.3073349, 63.0792656
4: -27.6933746, 41.6349754, -16.9646912, 25.6648083, -53.3581848, 58.5996666

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3901055, upper bound: 46.3859374
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4000627, upper bound: 46.4032096
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -13.1550112, 23.5543861, -46.7900314, 53.6009712
1: -26.5506458, 40.1011238, -15.1165361, 23.5508289, -50.1014748, 55.2176590
2: -26.2936344, 39.2237549, -14.9430695, 22.8703365, -49.1639671, 54.1668243
3: -33.3686562, 47.1178703, -19.2635593, 27.7262726, -61.0949211, 66.3814087
4: -29.7111588, 44.5652657, -17.2102947, 26.0118160, -55.7229691, 61.7755585

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3930251, upper bound: 46.3857680
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999862, upper bound: 46.4035650
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5350533, 37.8945465, -21.8700180, 38.4492760, -59.9295769, 59.7063408
1: -24.6167240, 37.5179062, -24.9983025, 38.0798302, -62.6965561, 62.5162086
2: -24.4222260, 36.7149124, -24.8030643, 37.2588768, -61.6304703, 61.4579620
3: -30.9534435, 44.0686035, -31.4313202, 44.7341652, -75.6876068, 75.4999161
4: -27.6933746, 41.6349754, -28.1271706, 42.2574234, -69.9452591, 69.7501526

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4031821, upper bound: 46.4104110
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028625, upper bound: 46.4078132
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.2356472, 40.4459610, -22.0251560, 38.7075157, -61.9129295, 62.4391747
1: -26.5506458, 40.1011238, -25.1752834, 38.3424797, -64.8931198, 65.2764053
2: -26.2936344, 39.2237549, -24.9815235, 37.5128365, -63.7848969, 64.1623306
3: -33.3686562, 47.1178703, -31.6525116, 45.0458679, -78.4145050, 78.7703552
4: -29.7111588, 44.5652657, -28.3318348, 42.5470848, -72.2582397, 72.8952866

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3416143, upper bound: 46.3234696
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4046490, upper bound: 46.4089856
time: 1.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3827630, upper bound: 46.3901055
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3827630, upper bound: 46.3975543
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3827630, upper bound: 46.3930749
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3827630, upper bound: 46.4000627
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.4035650, upper bound: 46.3999862
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3901055, upper bound: 46.3859374
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.4000627, upper bound: 46.4032096
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3930251, upper bound: 46.3857680
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3999862, upper bound: 46.4035650
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.4031821, upper bound: 46.4104110
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.4028625, upper bound: 46.4078132
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.3416143, upper bound: 46.3234696
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -46.4046490, upper bound: 46.4089856

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.9954882, 21.8092995, -12.2531643, 22.1069641, -34.1024475, 34.0624619
1: -13.8001146, 21.7692509, -14.0918055, 22.0766201, -35.8767319, 35.8610573
2: -13.6789618, 21.1440964, -13.9387560, 21.4467888, -35.1257477, 35.0828514
3: -17.6192856, 25.6233959, -17.9780922, 25.9761715, -43.5954590, 43.6014824
4: -15.8222933, 23.9692707, -16.0629044, 24.3521557, -40.1744461, 40.0321732

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3882196
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.1773834, 22.1378307, -13.8695087, 24.5627956, -36.7401810, 36.0073395
1: -14.0113878, 22.1042061, -15.9317598, 24.5727539, -38.5841408, 38.0359650
2: -13.8910999, 21.4651928, -15.7138424, 23.8699512, -37.7610474, 37.1790314
3: -17.8888302, 26.0218849, -20.3035431, 28.9352589, -46.8240776, 46.3254204
4: -16.0821590, 24.3427982, -18.0083065, 27.2013359, -43.2834930, 42.3511047

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3882196
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1616650, 23.6282253, -12.5226059, 22.4342117, -35.5958710, 36.1508331
1: -15.1181545, 23.5727386, -14.3898468, 22.4349022, -37.5530548, 37.9625816
2: -14.9563684, 22.9073143, -14.2071457, 21.7896976, -36.7460670, 37.1144600
3: -19.2627468, 27.7634983, -18.3484516, 26.4031906, -45.6659241, 46.1119385
4: -17.1972427, 26.0260620, -16.3346882, 24.7722683, -41.9695015, 42.3607483

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3357525, 23.9400272, -14.1164722, 24.8607445, -38.1964951, 38.0564995
1: -15.3200321, 23.8931084, -16.2042446, 24.9032745, -40.2233047, 40.0973511
2: -15.1585360, 23.2128086, -15.9665689, 24.1829910, -39.3415260, 39.1793785
3: -19.5203056, 28.1445408, -20.6430035, 29.3326225, -48.8529205, 48.7875443
4: -17.4483700, 26.3731804, -18.2583809, 27.5870838, -45.0354500, 44.6315613

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.9954882, 21.8092995, -21.0668125, 37.2215614, -49.2170486, 42.8761139
1: -13.8001146, 21.7692509, -24.0843430, 36.8374939, -50.6376038, 45.8535881
2: -13.6789618, 21.1440964, -23.9120750, 36.0505600, -49.7295151, 45.0561714
3: -17.6192856, 25.6233959, -30.2892323, 43.2680473, -60.8873329, 55.9126205
4: -15.8222933, 23.9692707, -27.1550350, 40.8504906, -56.6727829, 51.1243057

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3921855
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3930251
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.1773834, 22.1378307, -22.7664661, 39.7743149, -51.9516983, 44.9042969
1: -14.0113878, 22.1042061, -26.0173893, 39.4187469, -53.4301338, 48.1215935
2: -13.8910999, 21.4651928, -25.7768269, 38.5595741, -52.4506683, 47.2420158
3: -17.8888302, 26.0218849, -32.7056427, 46.3176727, -64.2065048, 58.7275238
4: -16.0821590, 24.3427982, -29.1707726, 43.7822456, -59.8644028, 53.5135727

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3921855
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3930251
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1616650, 23.6282253, -21.2301483, 37.3758888, -50.5375481, 44.8583679
1: -15.1181545, 23.5727386, -24.2700081, 37.0172615, -52.1354141, 47.8427391
2: -14.9563684, 22.9073143, -24.0754547, 36.2241287, -51.1804962, 46.9827652
3: -19.2627468, 27.7634983, -30.5272541, 43.4859314, -62.7486687, 58.2907448
4: -17.1972427, 26.0260620, -27.3000069, 41.0705757, -58.2678146, 53.3260651

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3999862
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3999862
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3357525, 23.9400272, -22.9142361, 39.9004250, -53.2361755, 46.8542633
1: -15.3200321, 23.8931084, -26.1856651, 39.5772514, -54.8972855, 50.0787735
2: -15.1585360, 23.2128086, -25.9322052, 38.7078133, -53.8663483, 49.1450081
3: -19.5203056, 28.1445408, -32.9182968, 46.5093269, -66.0296249, 61.0628357
4: -17.4483700, 26.3731804, -29.3014565, 43.9727211, -61.4210892, 55.6746292

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3999862
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4035651, upper bound: 46.3999862
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -21.0668125, 37.2215614, -11.9954882, 21.8092995, -42.8761139, 49.2170486
1: -24.0843430, 36.8374939, -13.8001146, 21.7692509, -45.8535919, 50.6376076
2: -23.9120750, 36.0505600, -13.6789618, 21.1440964, -45.0561714, 49.7295151
3: -30.2892323, 43.2680473, -17.6192856, 25.6233959, -55.9126167, 60.8873291
4: -27.1550350, 40.8504906, -15.8222933, 23.9692707, -51.1243057, 56.6727829

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3899620, upper bound: 46.3854112
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3851349, upper bound: 46.3844203
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.2301483, 37.3758888, -13.1613836, 23.6278553, -44.8580017, 50.5372734
1: -24.2700081, 37.0172615, -15.1178331, 23.5723267, -47.8423271, 52.1350937
2: -24.0754547, 36.2241287, -14.9560652, 22.9069443, -46.9823990, 51.1801910
3: -30.5272541, 43.4859314, -19.2623043, 27.7629890, -58.2902412, 62.7482224
4: -27.3000069, 41.0705757, -17.1968651, 26.0256042, -53.3255959, 58.2674408

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.3999549
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3908525
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.7664661, 39.7743149, -12.1773834, 22.1378307, -44.9042969, 51.9516983
1: -26.0173893, 39.4187469, -14.0113878, 22.1042061, -48.1215935, 53.4301338
2: -25.7768269, 38.5595741, -13.8910999, 21.4651928, -47.2420120, 52.4506645
3: -32.7056427, 46.3176727, -17.8888302, 26.0218849, -58.7275238, 64.2064896
4: -29.1707726, 43.7822456, -16.0821590, 24.3427982, -53.5135727, 59.8644028

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3883213, upper bound: 46.3852442
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3850690, upper bound: 46.3844658
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.9142361, 39.9004250, -13.3357525, 23.9400272, -46.8542633, 53.2361755
1: -26.1856651, 39.5772514, -15.3200321, 23.8931084, -50.0787735, 54.8972855
2: -25.9322052, 38.7078133, -15.1585360, 23.2128086, -49.1450081, 53.8663483
3: -32.9182968, 46.5093269, -19.5203056, 28.1445408, -61.0628357, 66.0296249
4: -29.3014565, 43.9727211, -17.4483700, 26.3731804, -55.6746292, 61.4210892

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4003102
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955037, upper bound: 46.4003102
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.2943287, 37.5009270, -21.8189716, 38.3776741, -59.6103477, 59.2629890
1: -24.3439159, 37.1416893, -24.9431095, 38.0891647, -62.4330788, 62.0848007
2: -24.1541214, 36.3430099, -24.7935638, 37.2426796, -61.3420181, 61.0575943
3: -30.6162510, 43.6292419, -31.3768730, 44.7672958, -75.3835449, 75.0061111
4: -27.4064884, 41.2113800, -28.1909790, 42.2644844, -69.6709671, 69.3997803

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5281677, 37.8836174, -21.6041851, 38.0325203, -59.5106735, 59.4270782
1: -24.6088543, 37.5069695, -24.6945267, 37.6634636, -62.2723122, 62.2014847
2: -24.4146557, 36.7041893, -24.5115414, 36.8500443, -61.2166023, 61.1529579
3: -30.9435692, 44.0557785, -31.0495510, 44.2463188, -75.1898880, 75.1053314
4: -27.6851463, 41.6228638, -27.8125420, 41.7928352, -69.4695587, 69.4243088

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3946316, upper bound: 46.4078132
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.7664661, 39.7743149, -20.9450912, 37.1654854, -59.8328056, 60.6408310
1: -26.0173893, 39.4187469, -23.9503860, 36.7825890, -62.7852821, 63.3691330
2: -25.7768269, 38.5595741, -23.8059731, 35.9914398, -61.6903229, 62.2734795
3: -32.7056427, 46.3176727, -30.1271534, 43.2099838, -75.9156189, 76.4448166
4: -29.1707726, 43.7822456, -27.0956306, 40.7520409, -69.8736954, 70.8215942

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3116623, upper bound: 46.3143690
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3411291, upper bound: 46.3223799
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.9142361, 39.9004250, -21.8168850, 38.4342804, -61.3204689, 61.7121010
1: -26.1856651, 39.5772514, -24.9382191, 38.0644608, -64.2501144, 64.5154724
2: -25.9322052, 38.7078133, -24.7670174, 37.2226715, -63.1267433, 63.4491615
3: -32.9182968, 46.5093269, -31.3724155, 44.7424889, -77.6607819, 77.8817368
4: -29.3014565, 43.9727211, -28.1194496, 42.2051010, -71.5065460, 72.0921631

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4042531, upper bound: 46.4051423
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4025087, upper bound: 46.4051423
time: 0.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3882196
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3882196
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3883213
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3947167, upper bound: 46.3947168
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3921855
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3930251
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3921855
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3930251
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3999862
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3999862
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823938, upper bound: 46.3999862
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.4035651, upper bound: 46.3999862
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3899620, upper bound: 46.3854112
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3851349, upper bound: 46.3844203
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3994915, upper bound: 46.3999549
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3908525
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3883213, upper bound: 46.3852442
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3850690, upper bound: 46.3844658
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4003102
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3955037, upper bound: 46.4003102
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3946316, upper bound: 46.4078132
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3116623, upper bound: 46.3143690
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3411291, upper bound: 46.3223799
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.4042531, upper bound: 46.4051423
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.4025087, upper bound: 46.4051423

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -12.2531643, 22.1069641, -33.8200684, 33.5707626
1: -13.4768963, 21.2664337, -14.0918055, 22.0766201, -35.5535164, 35.3582382
2: -13.3540964, 20.6594162, -13.9387560, 21.4467888, -34.8008842, 34.5981712
3: -17.2112961, 25.0253677, -17.9780922, 25.9761715, -43.1874695, 43.0034599
4: -15.4359884, 23.4152107, -16.0629044, 24.3521557, -39.7881432, 39.4781113

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3758764, upper bound: 46.3793668
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3758764, upper bound: 46.3793668
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -12.2531643, 22.1069641, -35.4042397, 35.9709244
1: -15.2800694, 23.7082329, -14.0918055, 22.0766201, -37.3566856, 37.8000374
2: -15.0858688, 23.0311298, -13.9387560, 21.4467888, -36.5326538, 36.9698868
3: -19.4841919, 27.9129372, -17.9780922, 25.9761715, -45.4603653, 45.8910255
4: -17.3320103, 26.2054291, -16.0629044, 24.3521557, -41.6841621, 42.2683258

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3758764, upper bound: 46.3793668
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3758764, upper bound: 46.3793668
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -13.8695087, 24.5627956, -36.2759056, 35.1871033
1: -13.4768963, 21.2664337, -15.9317598, 24.5727539, -38.0496521, 37.1981926
2: -13.3540964, 20.6594162, -15.7138424, 23.8699512, -37.2240486, 36.3732567
3: -17.2112961, 25.0253677, -20.3035431, 28.9352589, -46.1465492, 45.3289070
4: -15.4359884, 23.4152107, -18.0083065, 27.2013359, -42.6373215, 41.4235153

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3818606, upper bound: 46.3880878
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3848022
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -13.8695087, 24.5627956, -37.8600731, 37.5872726
1: -15.2800694, 23.7082329, -15.9317598, 24.5727539, -39.8528214, 39.6399918
2: -15.0858688, 23.0311298, -15.7138424, 23.8699512, -38.9558182, 38.7449722
3: -19.4841919, 27.9129372, -20.3035431, 28.9352589, -48.4194489, 48.2164726
4: -17.3320103, 26.2054291, -18.0083065, 27.2013359, -44.5333481, 44.2137375

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3818606, upper bound: 46.3883213
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3850690
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -12.5226059, 22.4342117, -35.2801208, 35.6099586
1: -14.7570267, 23.0229206, -14.3898468, 22.4349022, -37.1919289, 37.4127655
2: -14.5964413, 22.3779869, -14.2071457, 21.7896976, -36.3861389, 36.5851326
3: -18.8066769, 27.1103859, -18.3484516, 26.4031906, -45.2098541, 45.4588318
4: -16.7738895, 25.4232101, -16.3346882, 24.7722683, -41.5461578, 41.7578964

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882195, upper bound: 46.3868315
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882196, upper bound: 46.3971463
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -12.5226059, 22.4342117, -36.8680840, 38.0159454
1: -16.5651245, 25.4739952, -14.3898468, 22.4349022, -39.0000267, 39.8638420
2: -16.3321190, 24.7531395, -14.2071457, 21.7896976, -38.1218109, 38.9602814
3: -21.0962143, 30.0184078, -18.3484516, 26.4031906, -47.4993973, 48.3668518
4: -18.6865940, 28.2100029, -16.3346882, 24.7722683, -43.4588585, 44.5446930

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882195, upper bound: 46.3868315
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3882196, upper bound: 46.3971463
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -14.1164722, 24.8607445, -37.7066498, 37.2038231
1: -14.7570267, 23.0229206, -16.2042446, 24.9032745, -39.6603012, 39.2271652
2: -14.5964413, 22.3779869, -15.9665689, 24.1829910, -38.7794266, 38.3445549
3: -18.8066769, 27.1103859, -20.6430035, 29.3326225, -48.1392860, 47.7533875
4: -16.7738895, 25.4232101, -18.2583809, 27.5870838, -44.3609734, 43.6815872

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3914341, upper bound: 46.3941385
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908525, upper bound: 46.3908526
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4310522, 25.4894829, -14.1164722, 24.8607445, -39.2917938, 39.6059532
1: -16.5618916, 25.4696083, -16.2042446, 24.9032745, -41.4651642, 41.6738472
2: -16.3289795, 24.7492275, -15.9665689, 24.1829910, -40.5119629, 40.7157974
3: -21.0917263, 30.0130558, -20.6430035, 29.3326225, -50.4243355, 50.6560593
4: -18.6827087, 28.2050896, -18.2583809, 27.5870838, -46.2697906, 46.4634705

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3914341, upper bound: 46.3941385
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908525, upper bound: 46.3908526
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -21.0668125, 37.2215614, -48.9346695, 42.3844070
1: -13.4768963, 21.2664337, -24.0843430, 36.8374939, -50.3143921, 45.3507767
2: -13.3540964, 20.6594162, -23.9120750, 36.0505600, -49.4046478, 44.5714912
3: -17.2112961, 25.0253677, -30.2892323, 43.2680473, -60.4793434, 55.3145905
4: -15.4359884, 23.4152107, -27.1550350, 40.8504906, -56.2864799, 50.5702438

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3822271, upper bound: 46.3927237
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3849433
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -21.0668125, 37.2215614, -50.5188370, 44.7845726
1: -15.2800694, 23.7082329, -24.0843430, 36.8374939, -52.1175613, 47.7925758
2: -15.0858688, 23.0311298, -23.9120750, 36.0505600, -51.1364250, 46.9432068
3: -19.4841919, 27.9129372, -30.2892323, 43.2680473, -62.7522392, 58.2021561
4: -17.3320103, 26.2054291, -27.1550350, 40.8504906, -58.1825027, 53.3604660

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3822271, upper bound: 46.3930749
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3890487
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.7131090, 21.3175964, -22.7664661, 39.7743149, -51.4874191, 44.0840607
1: -13.4768963, 21.2664337, -26.0173893, 39.4187469, -52.8956451, 47.2838211
2: -13.3540964, 20.6594162, -25.7768269, 38.5595741, -51.9136658, 46.4362411
3: -17.2112961, 25.0253677, -32.7056427, 46.3176727, -63.5289688, 57.7310066
4: -15.4359884, 23.4152107, -29.1707726, 43.7822456, -59.2182350, 52.5859833

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3818606, upper bound: 46.3920625
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3881461
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2972765, 23.7177620, -22.7664661, 39.7743149, -53.0715904, 46.4842262
1: -15.2800694, 23.7082329, -26.0173893, 39.4187469, -54.6988144, 49.7256241
2: -15.0858688, 23.0311298, -25.7768269, 38.5595741, -53.6454430, 48.8079567
3: -19.4841919, 27.9129372, -32.7056427, 46.3176727, -65.8018646, 60.6185684
4: -17.3320103, 26.2054291, -29.1707726, 43.7822456, -61.1142578, 55.3762016

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3818606, upper bound: 46.3930251
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3844203, upper bound: 46.3890487
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -21.2301483, 37.3758888, -50.2217941, 44.3174896
1: -14.7570267, 23.0229206, -24.2700081, 37.0172615, -51.7742882, 47.2929268
2: -14.5964413, 22.3779869, -24.0754547, 36.2241287, -50.8205643, 46.4534416
3: -18.8066769, 27.1103859, -30.5272541, 43.4859314, -62.2926064, 57.6376343
4: -16.7738895, 25.4232101, -27.3000069, 41.0705757, -57.8444672, 52.7232094

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999550, upper bound: 46.3994915
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3953593
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4338799, 25.4933434, -21.2301483, 37.3758888, -51.8097649, 46.7234840
1: -16.5651245, 25.4739952, -24.2700081, 37.0172615, -53.5823860, 49.7439957
2: -16.3321190, 24.7531395, -24.0754547, 36.2241287, -52.5562401, 48.8285904
3: -21.0962143, 30.0184078, -30.5272541, 43.4859314, -64.5821381, 60.5456505
4: -18.6865940, 28.2100029, -27.3000069, 41.0705757, -59.7571716, 55.5100021

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999549, upper bound: 46.3994915
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3997371, upper bound: 46.3953593
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8459072, 23.0873528, -22.9142361, 39.9004250, -52.7463303, 46.0015869
1: -14.7570267, 23.0229206, -26.1856651, 39.5772514, -54.3342781, 49.2085876
2: -14.5964413, 22.3779869, -25.9322052, 38.7078133, -53.3042526, 48.3101883
3: -18.8066769, 27.1103859, -32.9182968, 46.5093269, -65.3160019, 60.0286827
4: -16.7738895, 25.4232101, -29.3014565, 43.9727211, -60.7466125, 54.7246628

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999550, upper bound: 46.3994150
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3997369, upper bound: 46.3955037
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4310522, 25.4894829, -22.9142361, 39.9004250, -54.3314743, 48.4037170
1: -16.5618916, 25.4696083, -26.1856651, 39.5772514, -56.1391449, 51.6552696
2: -16.3289795, 24.7492275, -25.9322052, 38.7078133, -55.0367928, 50.6814346
3: -21.0917263, 30.0130558, -32.9182968, 46.5093269, -67.6010437, 62.9313507
4: -18.6827087, 28.2050896, -29.3014565, 43.9727211, -62.6554298, 57.5065422

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999549, upper bound: 46.3994150
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3812547, upper bound: 46.3954939
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.0242786, 37.1601791, -11.7938385, 21.4853172, -42.5095940, 48.9540176
1: -24.0387039, 36.8581810, -13.5724163, 21.4605331, -45.4992371, 50.4305954
2: -23.9118500, 36.0451202, -13.4544125, 20.8393593, -44.7512093, 49.4995270
3: -30.2458897, 43.3149376, -17.3372669, 25.2613831, -55.5072708, 60.6521988
4: -27.2285213, 40.8701019, -15.5817976, 23.6190376, -50.8475456, 56.4519005

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3927237, upper bound: 46.3854112
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3899620, upper bound: 46.3854112
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -20.7976589, 36.8014526, -11.9894485, 21.7998600, -42.5975189, 48.7908974
1: -23.7767487, 36.4129028, -13.7931557, 21.7594776, -45.5362244, 50.2060585
2: -23.6167755, 35.6337662, -13.6723471, 21.1348495, -44.7516136, 49.3061104
3: -29.9028130, 42.7708855, -17.6105289, 25.6122322, -55.5150375, 60.3814163
4: -26.8352985, 40.3781166, -15.8150730, 23.9585533, -50.7938461, 56.1931915

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3849433, upper bound: 46.3844203
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3849433, upper bound: 46.3844203
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.1844101, 37.3096504, -12.9369373, 23.2689075, -44.4533081, 50.2465782
1: -24.2202110, 37.0313377, -14.8640709, 23.2277336, -47.4479446, 51.8954086
2: -24.0716362, 36.2114410, -14.7060871, 22.5676117, -46.6392479, 50.9175224
3: -30.4775429, 43.5230331, -18.9475365, 27.3587494, -57.8362923, 62.4705696
4: -27.3700848, 41.0786400, -16.9280968, 25.6327095, -53.0027885, 58.0067368

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.3999549
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994915, upper bound: 46.3999549
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.9624348, 36.9521980, -13.1552153, 23.6184196, -44.5808487, 50.1074104
1: -23.9639225, 36.5923882, -15.1107759, 23.5625477, -47.5264702, 51.7031593
2: -23.7815399, 35.8074646, -14.9493780, 22.8974228, -46.6789627, 50.7568436
3: -30.1430035, 42.9878731, -19.2534313, 27.7514954, -57.8945007, 62.2412987
4: -26.9810238, 40.5988998, -17.1895905, 26.0148754, -52.9958992, 57.7884903

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3908525
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3910784, upper bound: 46.3997369
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.7213936, 39.6971970, -11.9628582, 21.7902832, -44.5116768, 51.6600571
1: -25.9705620, 39.4170723, -13.7692089, 21.7724056, -47.7429657, 53.1862793
2: -25.7660904, 38.5337219, -13.6515341, 21.1375504, -46.9036407, 52.1852493
3: -32.6544037, 46.3389473, -17.5881958, 25.6327152, -58.2871132, 63.9271317
4: -29.2373409, 43.7738838, -15.8242769, 23.9673004, -53.2046394, 59.5981560

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880878, upper bound: 46.3852442
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880878, upper bound: 46.3852442
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.4876995, 39.3380928, -12.1713953, 22.1284943, -44.6161919, 51.5094872
1: -25.6985207, 38.9817810, -14.0044823, 22.0945473, -47.7930679, 52.9862633
2: -25.4682560, 38.1305084, -13.8845510, 21.4560528, -46.9243088, 52.0150528
3: -32.3065109, 45.8070488, -17.8801498, 26.0108643, -58.3173752, 63.6871910
4: -28.8407097, 43.2958755, -16.0750217, 24.3322201, -53.1729240, 59.3708878

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848022, upper bound: 46.3844658
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848022, upper bound: 46.3844658
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.8687057, 39.8262367, -13.1114569, 23.5816288, -46.4503326, 52.9376831
1: -26.1371765, 39.5747910, -15.0663137, 23.5491390, -49.6863174, 54.6411018
2: -25.9117527, 38.6826210, -14.9088440, 22.8739834, -48.7857361, 53.5914650
3: -32.8669434, 46.5294876, -19.2056828, 27.7410927, -60.6080322, 65.7351685
4: -29.3627548, 43.9634056, -17.1799908, 25.9814606, -55.3442078, 61.1433945

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4003102
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994150, upper bound: 46.4003102
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.6378059, 39.4662666, -13.3295307, 23.9305477, -46.5683517, 52.7957993
1: -25.8694973, 39.1407585, -15.3128815, 23.8832855, -49.7527733, 54.4536400
2: -25.6257515, 38.2808342, -15.1517820, 23.2032433, -48.8289948, 53.4326057
3: -32.5233040, 46.0003319, -19.5113659, 28.1329937, -60.6562920, 65.5116730
4: -28.9722366, 43.4896584, -17.4409924, 26.3623905, -55.3346214, 60.9306488

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3908526, upper bound: 46.4003092
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3955037, upper bound: 46.4003102
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -21.8189716, 38.3776741, -59.8174934, 59.5897369
1: -24.5586433, 37.5217667, -24.9431095, 38.0891647, -62.6478081, 62.4648743
2: -24.4099751, 36.6918221, -24.7935638, 37.2426796, -61.5928230, 61.4151039
3: -30.8934937, 44.0957451, -31.3768730, 44.7672958, -75.6607819, 75.4726181
4: -27.7538719, 41.6327591, -28.1909790, 42.2644844, -70.0183563, 69.8237381

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4031600, upper bound: 46.4104109
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953312, upper bound: 46.3976575
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -21.8189716, 38.3776741, -59.5768890, 59.2285004
1: -24.3087234, 37.0905914, -24.9431095, 38.0891647, -62.3978844, 62.0336990
2: -24.1263924, 36.2956047, -24.7935638, 37.2426796, -61.3093338, 61.0093727
3: -30.5667114, 43.5680695, -31.3768730, 44.7672958, -75.3340073, 74.9449387
4: -27.3722706, 41.1605835, -28.1909790, 42.2644844, -69.6367416, 69.3452072

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953312, upper bound: 46.3976575
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953312, upper bound: 46.3976575
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.4818611, 37.8172760, -21.6041851, 38.0325203, -59.4605255, 59.3507042
1: -24.5586433, 37.5217667, -24.6945267, 37.6634636, -62.2221069, 62.2162933
2: -24.4099751, 36.6918221, -24.5115414, 36.8500443, -61.1900063, 61.1333618
3: -30.8934937, 44.0957451, -31.0495510, 44.2463188, -75.1397934, 75.1452942
4: -27.7538719, 41.6327591, -27.8125420, 41.7928352, -69.5453339, 69.4423447

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4026640, upper bound: 46.4064392
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4026641, upper bound: 46.4064392
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.2656136, 37.4670982, -21.6041851, 38.0325203, -59.2457886, 59.0157585
1: -24.3087234, 37.0905914, -24.6945267, 37.6634636, -61.9721794, 61.7851181
2: -24.1263924, 36.2956047, -24.5115414, 36.8500443, -60.9258003, 60.7472839
3: -30.5667114, 43.5680695, -31.0495510, 44.2463188, -74.8130264, 74.6176224
4: -27.3722706, 41.1605835, -27.8125420, 41.7928352, -69.1574936, 68.9597092

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3946316, upper bound: 46.4064392
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3946316, upper bound: 46.3946316
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.4566593, 39.3044662, -20.9225121, 37.1307144, -59.4892960, 60.1471977
1: -25.6682167, 38.9601212, -23.9250240, 36.7489204, -62.4042473, 62.8851357
2: -25.4331322, 38.1066360, -23.7811069, 35.9580460, -61.3156357, 61.7941856
3: -32.2738647, 45.7820358, -30.0959969, 43.1706810, -75.4445496, 75.8780289
4: -28.8113766, 43.2537880, -27.0694485, 40.7134361, -69.4764099, 70.2655258

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3411291, upper bound: 46.3223799
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=55.00176239013672
rel_dist={3: [-46.41123857032521, 46.41123857032903]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1119.67 seconds
