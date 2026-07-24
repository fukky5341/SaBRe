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
execution time: IAR + LP analysis = 2.61 + 1.90 = 4.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -46.4113886, upper bound: 46.4113886


# Binary Search by BASE starts (time budget: 1195.49 seconds, max iter: 100)

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
Binary search time: 77.77 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1117.73 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
time: 0.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4113240, upper bound: 46.4113238
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 3, lower bound: -46.4113240, upper bound: 46.4113238

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -15.6748066, 27.2767467, -40.5789375, 39.4772720
1: -15.2836475, 23.8005772, -17.9646091, 27.3548412, -42.6384850, 41.7651787
2: -15.1096706, 23.1136723, -17.7045784, 26.5651569, -41.6748276, 40.8182526
3: -19.4732094, 28.0227852, -22.8100491, 32.1917152, -51.6649170, 50.8328323
4: -17.4002285, 26.2846336, -20.1704769, 30.4162006, -47.8164215, 46.4551048

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

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
time: 1.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -15.6724682, 27.2734356, -49.4864922, 54.6902313
1: -25.3891716, 38.6557007, -17.9620037, 27.3516922, -52.7408638, 56.6177063
2: -25.1929913, 37.8162651, -17.7019806, 26.5621128, -51.7550964, 55.5182457
3: -31.9208298, 45.4162788, -22.8069592, 32.1879921, -64.1088257, 68.2232361
4: -28.5711136, 42.8956490, -20.1677132, 30.4125576, -58.9836731, 63.0633621

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

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
- Time for IS candidates: 4.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4032405
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113240

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.3021946, 23.8024693, -37.1046600, 37.1046600
1: -15.2836475, 23.8005772, -15.2836475, 23.8005772, -39.0842171, 39.0842209
2: -15.1096706, 23.1136723, -15.1096706, 23.1136723, -38.2233429, 38.2233429
3: -19.4732094, 28.0227852, -19.4732094, 28.0227852, -47.4959869, 47.4959869
4: -17.4002285, 26.2846336, -17.4002285, 26.2846336, -43.6848602, 43.6848602

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3784407
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
time: 0.75 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -22.2130566, 39.0177612, -52.3199539, 46.0155220
1: -15.2836475, 23.8005772, -25.3891716, 38.6557007, -53.9393463, 49.1897392
2: -15.1096706, 23.1136723, -25.1929913, 37.8162651, -52.9259338, 48.3066559
3: -19.4732094, 28.0227852, -31.9208298, 45.4162788, -64.8894882, 59.9436111
4: -17.4002285, 26.2846336, -28.5711136, 42.8956490, -60.2958755, 54.8557472

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3784407
time: 1.05 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
time: 0.77 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -13.3021946, 23.8024693, -46.0155220, 52.3199539
1: -25.3891716, 38.6557007, -15.2836475, 23.8005772, -49.1897392, 53.9393463
2: -25.1929913, 37.8162651, -15.1096706, 23.1136723, -48.3066559, 52.9259338
3: -31.9208298, 45.4162788, -19.4732094, 28.0227852, -59.9436111, 64.8894806
4: -28.5711136, 42.8956490, -17.4002285, 26.2846336, -54.8557472, 60.2958755

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
time: 0.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -22.2130566, 39.0177612, -61.2015877, 61.2015839
1: -25.3891716, 38.6557007, -25.3891716, 38.6557007, -64.0448761, 64.0448761
2: -25.1929913, 37.8162651, -25.1929913, 37.8162651, -62.9735107, 62.9735184
3: -31.9208298, 45.4162788, -31.9208298, 45.4162788, -77.3370972, 77.3370972
4: -28.5711136, 42.8956490, -28.5711136, 42.8956490, -71.4667664, 71.4667664

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784407, upper bound: 46.3966428
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4025283, upper bound: 46.4087713
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.14 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3784407
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3784407
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3784407, upper bound: 46.3966428
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.4025283, upper bound: 46.4087713

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -11.3869448, 20.9255638, -34.2277527, 35.1894150
1: -15.2836475, 23.8005772, -13.0953932, 20.8397293, -36.1233749, 36.8959579
2: -15.1096706, 23.1136723, -13.0110359, 20.2390823, -35.3487549, 36.1247101
3: -19.4732094, 28.0227852, -16.7173176, 24.5306568, -44.0038528, 44.7401009
4: -17.4002285, 26.2846336, -15.0642700, 22.9335842, -40.3338127, 41.3489037

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3885100, upper bound: 46.3473082
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.0015049, 23.3439465, -36.6461411, 36.8039742
1: -15.2836475, 23.8005772, -14.9456024, 23.3515148, -38.6351547, 38.7461777
2: -15.1096706, 23.1136723, -14.7805185, 22.6720467, -37.7817154, 37.8941917
3: -19.4732094, 28.0227852, -19.0547180, 27.4934063, -46.9666100, 47.0775032
4: -17.4002285, 26.2846336, -17.0490189, 25.7683182, -43.1685410, 43.3336525

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3763450, upper bound: 46.3951509
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -20.1475525, 35.8621483, -49.1643410, 43.9500198
1: -15.2836475, 23.8005772, -23.0306740, 35.4290886, -50.7127342, 46.8312416
2: -15.1096706, 23.1136723, -22.9083958, 34.6756935, -49.7853622, 46.0220680
3: -19.4732094, 28.0227852, -28.9547653, 41.6189651, -61.0921669, 56.9775505
4: -17.4002285, 26.2846336, -26.0378189, 39.2533302, -56.6535568, 52.3224525

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4034661, upper bound: 46.3517493
time: 1.01 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4071149, upper bound: 46.3814746
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -21.9027481, 38.5440521, -51.8462448, 45.7052155
1: -15.2836475, 23.8005772, -25.0402546, 38.1956673, -53.4793167, 48.8408279
2: -15.1096706, 23.1136723, -24.8525200, 37.3617020, -52.4713745, 47.9661942
3: -19.4732094, 28.0227852, -31.4891014, 44.8785400, -64.3517456, 59.5118866
4: -17.4002285, 26.2846336, -28.2129040, 42.3660202, -59.7662506, 54.4975357

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4037744, upper bound: 46.4008411
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4037744, upper bound: 46.3978337
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -13.3021946, 23.8024693, -43.9500198, 49.1643410
1: -23.0306740, 35.4290886, -15.2836475, 23.8005772, -46.8312416, 50.7127342
2: -22.9083958, 34.6756935, -15.1096706, 23.1136723, -46.0220680, 49.7853622
3: -28.9547653, 41.6189651, -19.4732094, 28.0227852, -56.9775505, 61.0921669
4: -26.0378189, 39.2533302, -17.4002285, 26.2846336, -52.3224525, 56.6535568

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3281423, upper bound: 46.4034657
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -13.3021946, 23.8024693, -45.7052155, 51.8462448
1: -25.0402546, 38.1956673, -15.2836475, 23.8005772, -48.8408241, 53.4793167
2: -24.8525200, 37.3617020, -15.1096706, 23.1136723, -47.9661942, 52.4713745
3: -31.4891014, 44.8785400, -19.4732094, 28.0227852, -59.5118866, 64.3517456
4: -28.2129040, 42.3660202, -17.4002285, 26.2846336, -54.4975357, 59.7662506

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4008411, upper bound: 46.4037744
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3978337, upper bound: 46.4037744
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -22.2130566, 39.0177612, -59.1370125, 57.9924889
1: -23.0306740, 35.4290886, -25.3891716, 38.6557007, -61.6863747, 60.8182602
2: -22.9083958, 34.6756935, -25.1929913, 37.8162651, -60.6736412, 59.8156548
3: -28.9547653, 41.6189651, -31.9208298, 45.4162788, -74.3710175, 73.5397873
4: -26.0378189, 39.2533302, -28.5711136, 42.8956490, -68.9334717, 67.8180923

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3828897, upper bound: 46.3882241
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3860705, upper bound: 46.4089608
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -22.2130566, 39.0177612, -60.8921165, 60.7265244
1: -25.0402546, 38.1956673, -25.3891716, 38.6557007, -63.6959534, 63.5848389
2: -24.8525200, 37.3617020, -25.1929913, 37.8162651, -62.6320419, 62.5175209
3: -31.4891014, 44.8785400, -31.9208298, 45.4162788, -76.9053802, 76.7993622
4: -28.2129040, 42.3660202, -28.5711136, 42.8956490, -71.1085510, 70.9371338

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4035493, upper bound: 46.3898807
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068176, upper bound: 46.4106174
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.15 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3885100, upper bound: 46.3473082
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3763450, upper bound: 46.3951509
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4034661, upper bound: 46.3517493
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4071149, upper bound: 46.3814746
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4037744, upper bound: 46.4008411
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4037744, upper bound: 46.3978337
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3281423, upper bound: 46.4034657
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4008411, upper bound: 46.4037744
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3978337, upper bound: 46.4037744
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3828897, upper bound: 46.3882241
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.3860705, upper bound: 46.4089608
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4035493, upper bound: 46.3898807
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -46.4068176, upper bound: 46.4106174

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -11.2742805, 20.7554340, -34.0576286, 35.0767517
1: -15.2836475, 23.8005772, -12.9542065, 20.7355442, -36.0191917, 36.7547836
2: -15.1096706, 23.1136723, -12.9108934, 20.1249809, -35.2346497, 36.0245667
3: -19.4732094, 28.0227852, -16.5860004, 24.4271832, -43.9003868, 44.6087875
4: -17.4002285, 26.2846336, -15.0160170, 22.8133278, -40.2135544, 41.3006516

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3286004, upper bound: 46.3181496
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3885100, upper bound: 46.3473082
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -11.2217398, 20.6840553, -33.9862404, 35.0242081
1: -15.2836475, 23.8005772, -12.9060535, 20.5976448, -35.8812904, 36.7066231
2: -15.1096706, 23.1136723, -12.8317413, 20.0037327, -35.1134033, 35.9454117
3: -19.4732094, 28.0227852, -16.4788857, 24.2461224, -43.7193298, 44.5016708
4: -17.4002285, 26.2846336, -14.8719177, 22.6551094, -40.0553360, 41.1565514

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3829983, upper bound: 46.3264551
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -12.9726534, 23.3038578, -35.1841736, 34.4733582
1: -13.6646070, 21.4798603, -14.9130402, 23.3105755, -36.9751778, 36.3928986
2: -13.4940720, 20.8352280, -14.7496967, 22.6321106, -36.1261826, 35.5849190
3: -17.4712029, 25.2716560, -19.0143070, 27.4450321, -44.9162331, 44.2859573
4: -15.5687332, 23.6612968, -17.0166626, 25.7211399, -41.2898712, 40.6779556

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3286004, upper bound: 46.3181496
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3746578, upper bound: 46.3901540
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -13.0015049, 23.3439465, -36.5361595, 36.6306534
1: -15.1584854, 23.6275005, -14.9456024, 23.3515148, -38.5100021, 38.5731049
2: -14.9872761, 22.9454288, -14.7805185, 22.6720467, -37.6593246, 37.7259483
3: -19.3160419, 27.8168468, -19.0547180, 27.4934063, -46.8094482, 46.8715630
4: -17.2645493, 26.0874233, -17.0490189, 25.7683182, -43.0328636, 43.1364441

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3956856, upper bound: 46.3989951
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3939982
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -13.2730513, 23.7619305, -18.5107365, 33.2032585, -46.4763031, 42.2726593
1: -15.2507362, 23.7590637, -21.1606407, 32.7483139, -47.9990501, 44.9197006
2: -15.0785065, 23.0732307, -21.0287209, 32.0600052, -47.1385117, 44.1019440
3: -19.4322109, 27.9736958, -26.6253719, 38.4386711, -57.8708801, 54.5990677
4: -17.3673306, 26.2367344, -23.9056454, 36.2302704, -53.5976028, 50.1423798

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4034657, upper bound: 46.3517490
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985053, upper bound: 46.3500619
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -20.0294838, 35.6806297, -48.9275627, 43.8319473
1: -15.2836475, 23.8005772, -22.8956375, 35.2464561, -50.5300941, 46.6962128
2: -15.1096706, 23.1136723, -22.7774448, 34.4977379, -49.6074066, 45.8911171
3: -19.4732094, 28.0227852, -28.7865505, 41.4025917, -60.8757896, 56.8093338
4: -17.4002285, 26.2846336, -25.8944168, 39.0450287, -56.4452553, 52.1790504

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4056535, upper bound: 46.3785322
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4058278, upper bound: 46.3789785
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -21.8556671, 38.4770317, -51.7792282, 45.6581345
1: -15.2836475, 23.8005772, -24.9799175, 38.2082138, -53.4918594, 48.7804871
2: -15.1096706, 23.1136723, -24.8422871, 37.3502884, -52.4599571, 47.9559593
3: -19.4732094, 28.0227852, -31.4298725, 44.9164658, -64.3896790, 59.4526596
4: -17.4002285, 26.2846336, -28.2795525, 42.3808212, -59.7810440, 54.5641861

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015520, upper bound: 46.3986698
time: 0.90 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996692, upper bound: 46.3984722
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -21.6283913, 38.1185913, -51.4207802, 45.4308548
1: -15.2836475, 23.8005772, -24.7276039, 37.7699165, -53.0535622, 48.5281715
2: -15.1096706, 23.1136723, -24.5516720, 36.9445229, -52.0541916, 47.6653442
3: -19.4732094, 28.0227852, -31.0978374, 44.3794708, -63.8526726, 59.1206169
4: -17.4002285, 26.2846336, -27.8886814, 41.8913345, -59.2915611, 54.1733170

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015464, upper bound: 46.3926129
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996636, upper bound: 46.3924153
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -18.5107365, 33.2032585, -13.2730513, 23.7619305, -42.2726593, 46.4763031
1: -21.1606407, 32.7483139, -15.2507362, 23.7590637, -44.9197006, 47.9990501
2: -21.0287209, 32.0600052, -15.0785065, 23.0732307, -44.1019440, 47.1385117
3: -26.6253719, 38.4386711, -19.4322109, 27.9736958, -54.5990677, 57.8708801
4: -23.9056454, 36.2302704, -17.3673306, 26.2367344, -50.1423798, 53.5976028

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517490, upper bound: 46.4034657
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3264551, upper bound: 46.3829983
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -20.0294838, 35.6806297, -13.3021946, 23.8024693, -43.8319511, 48.9275627
1: -22.8956375, 35.2464561, -15.2836475, 23.8005772, -46.6962128, 50.5300941
2: -22.7774448, 34.4977379, -15.1096706, 23.1136723, -45.8911171, 49.6074066
3: -28.7865505, 41.4025917, -19.4732094, 28.0227852, -56.8093338, 60.8757896
4: -25.8944168, 39.0450287, -17.4002285, 26.2846336, -52.1790504, 56.4452553

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3785322, upper bound: 46.4056535
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3789785, upper bound: 46.4058276
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -13.3021946, 23.8024693, -45.6581345, 51.7792282
1: -24.9799175, 38.2082138, -15.2836475, 23.8005772, -48.7804871, 53.4918594
2: -24.8422871, 37.3502884, -15.1096706, 23.1136723, -47.9559593, 52.4599571
3: -31.4298725, 44.9164658, -19.4732094, 28.0227852, -59.4526596, 64.3896790
4: -28.2795525, 42.3808212, -17.4002285, 26.2846336, -54.5641861, 59.7810440

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986698, upper bound: 46.4015520
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3984722, upper bound: 46.3996692
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -13.3021946, 23.8024693, -45.4308586, 51.4207802
1: -24.7276039, 37.7699165, -15.2836475, 23.8005772, -48.5281715, 53.0535622
2: -24.5516720, 36.9445229, -15.1096706, 23.1136723, -47.6653442, 52.0541916
3: -31.0978374, 44.3794708, -19.4732094, 28.0227852, -59.1206169, 63.8526764
4: -27.8886814, 41.8913345, -17.4002285, 26.2846336, -54.1733170, 59.2915611

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3926129, upper bound: 46.4015464
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3924153, upper bound: 46.3996636
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -20.1171017, 35.8186760, -20.6246643, 36.4598656, -56.5333252, 56.3004417
1: -22.9960213, 35.3854179, -23.5747108, 36.0849686, -59.0809860, 58.9208565
2: -22.8755836, 34.6329155, -23.3903923, 35.2937012, -58.1063881, 57.9376755
3: -28.9119282, 41.5676460, -29.6732693, 42.3714294, -71.2833557, 71.2409058
4: -26.0034389, 39.2031708, -26.5407066, 39.9774590, -65.9785995, 65.7260590

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3712096, upper bound: 46.3714470
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -22.0927143, 38.8323898, -58.8867493, 57.8491325
1: -23.0306740, 35.4290886, -25.2516708, 38.4694748, -61.4999504, 60.6807556
2: -22.9083958, 34.6756935, -25.0594330, 37.6350479, -60.4278336, 59.6613655
3: -28.9547653, 41.6189651, -31.7492695, 45.1954880, -74.1502380, 73.3682098
4: -26.0378189, 39.2533302, -28.4246864, 42.6831512, -68.6812439, 67.6318741

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3571279, upper bound: 46.3998272
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3857363, upper bound: 46.4059533
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.8720856, 38.5003624, -20.6246643, 36.4598656, -58.2882538, 59.0341988
1: -25.0054703, 38.1516800, -23.5747108, 36.0849686, -61.0904388, 61.7026749
2: -24.8194141, 37.3186531, -23.3903923, 35.2937012, -60.0645294, 60.6392441
3: -31.4459667, 44.8268471, -29.6732693, 42.3714294, -73.8173828, 74.5001068
4: -28.1782799, 42.3154488, -26.5407066, 39.9774590, -68.1557388, 68.8561554

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4001282, upper bound: 46.3885531
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4035493, upper bound: 46.3898807
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4024553, upper bound: 46.3898807
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -22.0927143, 38.8323898, -60.6418533, 60.5831680
1: -25.0402546, 38.1956673, -25.2516708, 38.4694748, -63.5036011, 63.4473381
2: -24.8525200, 37.3617020, -25.0594330, 37.6350479, -62.3862419, 62.3632317
3: -31.4891014, 44.8785400, -31.7492695, 45.1954880, -76.6845856, 76.6277924
4: -28.2129040, 42.3660202, -28.4246864, 42.6831512, -70.8618546, 70.7627563

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4049882, upper bound: 46.4086192
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4038438, upper bound: 46.4080401
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.48 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3286004, upper bound: 46.3181496
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3885100, upper bound: 46.3473082
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3829983, upper bound: 46.3264551
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3286004, upper bound: 46.3181496
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3746578, upper bound: 46.3901540
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3956856, upper bound: 46.3989951
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3939982
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4034657, upper bound: 46.3517490
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3985053, upper bound: 46.3500619
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4056535, upper bound: 46.3785322
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4058278, upper bound: 46.3789785
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4015520, upper bound: 46.3986698
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3996692, upper bound: 46.3984722
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4015464, upper bound: 46.3926129
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3996636, upper bound: 46.3924153
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3517490, upper bound: 46.4034657
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3264551, upper bound: 46.3829983
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3785322, upper bound: 46.4056535
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3789785, upper bound: 46.4058276
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3986698, upper bound: 46.4015520
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3984722, upper bound: 46.3996692
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3926129, upper bound: 46.4015464
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3924153, upper bound: 46.3996636
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3571279, upper bound: 46.3998272
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.3857363, upper bound: 46.4059533
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4035493, upper bound: 46.3898807
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4024553, upper bound: 46.3898807
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4049882, upper bound: 46.4086192
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 3, lower bound: -46.4038438, upper bound: 46.4080401

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -11.2501421, 20.7195435, -32.5998611, 32.7508469
1: -13.6646070, 21.4798603, -12.9263668, 20.6993523, -34.3639603, 34.4062233
2: -13.4940720, 20.8352280, -12.8840065, 20.0900364, -33.5841064, 33.7192345
3: -17.4712029, 25.2716560, -16.5517349, 24.3846416, -41.8558426, 41.8233833
4: -15.5687332, 23.6612968, -14.9874001, 22.7717705, -38.3405037, 38.6486893

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3263204, upper bound: 46.3167829
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3264790, upper bound: 46.3143040
time: 0.93 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3226822, upper bound: 46.3181496
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3017873, upper bound: 46.3169766
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3280825, upper bound: 46.3168242
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2968855, upper bound: 46.3181496
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2968855, upper bound: 46.3181496
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -11.2742805, 20.7554340, -33.9476471, 34.9034271
1: -15.1584854, 23.6275005, -12.9542065, 20.7355442, -35.8940277, 36.5817070
2: -14.9872761, 22.9454288, -12.9108934, 20.1249809, -35.1122589, 35.8563194
3: -19.3160419, 27.8168468, -16.5860004, 24.4271832, -43.7432251, 44.4028473
4: -17.2645493, 26.0874233, -15.0160170, 22.8133278, -40.0778770, 41.1034393

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3868266, upper bound: 46.3460774
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3849438, upper bound: 46.3458797
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.2730513, 23.7619305, -9.8099222, 18.3955498, -31.6686001, 33.5718422
1: -15.2507362, 23.7590637, -11.2973719, 18.3045788, -33.5553131, 35.0564308
2: -15.0785065, 23.0732307, -11.2150269, 17.7656765, -32.8441849, 34.2882538
3: -19.4322109, 27.9736958, -14.4795485, 21.5189934, -40.9512024, 42.4532433
4: -17.3673306, 26.2367344, -13.0337639, 20.0603638, -37.4276962, 39.2705002

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3697266, upper bound: 46.3188755
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3785013, upper bound: 46.3245632
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -11.1346703, 20.5469704, -33.8491592, 34.9371338
1: -15.2836475, 23.8005772, -12.8063316, 20.4618702, -35.7455177, 36.6069069
2: -15.1096706, 23.1136723, -12.7347527, 19.8716698, -34.9813385, 35.8484230
3: -19.4732094, 28.0227852, -16.3541832, 24.0840511, -43.5572510, 44.3769646
4: -17.4002285, 26.2846336, -14.7644615, 22.4989681, -39.8991966, 41.0490952

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3783742, upper bound: 46.3683297
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3871489, upper bound: 46.3740174
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -12.7953711, 23.1070862, -34.9874039, 34.2960739
1: -13.6646070, 21.4798603, -14.7180500, 23.1747131, -36.8393211, 36.1979103
2: -13.4940720, 20.8352280, -14.6152344, 22.4790668, -35.9731369, 35.4504585
3: -17.4712029, 25.2716560, -18.7839432, 27.3015556, -44.7727547, 44.0555992
4: -15.5687332, 23.6612968, -16.9532089, 25.5573559, -41.1260910, 40.6145020

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3754509, upper bound: 46.3939792
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3642384, upper bound: 46.3820772
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3716510, upper bound: 46.3923321
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -12.7212696, 22.9188881, -34.7992096, 34.2219772
1: -13.6646070, 21.4798603, -14.6251431, 22.9130669, -36.5776749, 36.1050034
2: -13.4940720, 20.8352280, -14.4763994, 22.2441349, -35.7382050, 35.3116264
3: -17.4712029, 25.2716560, -18.6537132, 26.9812775, -44.4524765, 43.9253654
4: -15.5687332, 23.6612968, -16.7179832, 25.2826214, -40.8513527, 40.3792763

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3735498, upper bound: 46.3866264
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3708134
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3901540
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -12.8240356, 23.1475544, -36.3397675, 36.4531822
1: -15.1584854, 23.6275005, -14.7506380, 23.2159920, -38.3744774, 38.3781395
2: -14.9872761, 22.9454288, -14.6458588, 22.5191956, -37.5064697, 37.5912857
3: -19.3160419, 27.8168468, -18.8241386, 27.3499374, -46.6659775, 46.6409836
4: -17.2645493, 26.0874233, -16.9858208, 25.6044750, -42.8690262, 43.0732422

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3915564, upper bound: 46.3972800
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -12.7501745, 22.9591522, -36.1513634, 36.3793221
1: -15.1584854, 23.6275005, -14.6577673, 22.9542313, -38.1127167, 38.2852669
2: -14.9872761, 22.9454288, -14.5072327, 22.2842579, -37.2715340, 37.4526596
3: -19.3160419, 27.8168468, -18.6941471, 27.0294781, -46.3455200, 46.5109901
4: -17.2645493, 26.0874233, -16.7502651, 25.3299522, -42.5945015, 42.8376884

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3919843, upper bound: 46.3902990
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3901015, upper bound: 46.3901013
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.0901480, 23.5535679, -18.5107365, 33.2032585, -46.2934074, 42.0643005
1: -15.0510550, 23.6126156, -21.1606407, 32.7483139, -47.7993698, 44.7732544
2: -14.9348707, 22.9084454, -21.0287209, 32.0600052, -46.9948769, 43.9371643
3: -19.1946678, 27.8182869, -26.6253719, 38.4386711, -57.6333389, 54.4436569
4: -17.2951431, 26.0583858, -23.9056454, 36.2302704, -53.5254097, 49.9640274

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4022238, upper bound: 46.3496487
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3902425, upper bound: 46.3429712
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3851667, upper bound: 46.3262504
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.0310068, 23.3925934, -18.5107365, 33.2032585, -46.2342644, 41.9033203
1: -14.9737244, 23.3782749, -21.1606407, 32.7483139, -47.7220383, 44.5389137
2: -14.8151503, 22.7017288, -21.0287209, 32.0600052, -46.8751564, 43.7304420
3: -19.0846157, 27.5269566, -26.6253719, 38.4386711, -57.5232849, 54.1523285
4: -17.0797577, 25.8154736, -23.9056454, 36.2302704, -53.3100243, 49.7211151

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3792955, upper bound: 46.3477472
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3852336, upper bound: 46.3430090
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3940083, upper bound: 46.3486967
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -12.8695917, 23.1978874, -20.0294838, 35.6806297, -48.4930000, 43.2273674
1: -14.8005753, 23.1950188, -22.8956375, 35.2464561, -50.0470238, 46.0906563
2: -14.6490326, 22.5233936, -22.7774448, 34.4977379, -49.1467705, 45.3008385
3: -18.8776188, 27.3054333, -28.7865505, 41.4025917, -60.2802086, 56.0919724
4: -16.9203701, 25.5841904, -25.8944168, 39.0450287, -55.9654007, 51.4786034

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3943928, upper bound: 46.3494641
time: 1.02 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4003575, upper bound: 46.3772750
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -12.2885456, 22.1907845, -20.0294838, 35.6806297, -47.8693314, 42.2202682
1: -14.1313658, 22.1690311, -22.8956375, 35.2464561, -49.3672028, 45.0646667
2: -13.9499454, 21.5208340, -22.7774448, 34.4977379, -48.4476852, 44.2982750
3: -18.0284634, 26.0830059, -28.7865505, 41.4025917, -59.4310493, 54.8695488
4: -16.0642853, 24.4581051, -25.8944168, 39.0450287, -55.1093140, 50.3525162

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3925099, upper bound: 46.3492664
time: 0.98 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3984747, upper bound: 46.3770773
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8695917, 23.1978874, -21.8556671, 38.4770317, -51.3466187, 45.0535507
1: -14.8005753, 23.1950188, -24.9799175, 38.2082138, -53.0087852, 48.1749344
2: -14.6490326, 22.5233936, -24.8422871, 37.3502884, -51.9993172, 47.3656807
3: -18.8776188, 27.3054333, -31.4298725, 44.9164658, -63.7940826, 58.7353020
4: -16.9203701, 25.5841904, -28.2795525, 42.3808212, -59.3011932, 53.8637428

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3831175, upper bound: 46.3949972
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015520, upper bound: 46.3986698
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.2885456, 22.1907845, -21.8556671, 38.4770317, -50.7655792, 44.0464516
1: -14.1313658, 22.1690311, -24.9799175, 38.2082138, -52.3395729, 47.1489410
2: -13.9499454, 21.5208340, -24.8422871, 37.3502884, -51.3002319, 46.3631172
3: -18.0284634, 26.0830059, -31.4298725, 44.9164658, -62.9449310, 57.5128746
4: -16.0642853, 24.4581051, -28.2795525, 42.3808212, -58.4451065, 52.7376556

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806663, upper bound: 46.3872477
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3982980, upper bound: 46.3428982
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8695917, 23.1978874, -21.6283913, 38.1185913, -50.9881783, 44.8262711
1: -14.8005753, 23.1950188, -24.7276039, 37.7699165, -52.5704880, 47.9226227
2: -14.6490326, 22.5233936, -24.5516720, 36.9445229, -51.5935555, 47.0750656
3: -18.8776188, 27.3054333, -31.0978374, 44.3794708, -63.2570877, 58.4032555
4: -16.9203701, 25.5841904, -27.8886814, 41.8913345, -58.8117027, 53.4728699

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3831120, upper bound: 46.3889403
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015466, upper bound: 46.3926129
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.2885456, 22.1907845, -21.6283913, 38.1185913, -50.4071350, 43.8191757
1: -14.1313658, 22.1690311, -24.7276039, 37.7699165, -51.9012794, 46.8966293
2: -13.9499454, 21.5208340, -24.5516720, 36.9445229, -50.8944664, 46.0725021
3: -18.0284634, 26.0830059, -31.0978374, 44.3794708, -62.4079323, 57.1808357
4: -16.0642853, 24.4581051, -27.8886814, 41.8913345, -57.9556198, 52.3467865

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993558, upper bound: 46.3889137
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3829091, upper bound: 46.3848179
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -18.5107365, 33.2032585, -13.0901480, 23.5535679, -42.0643005, 46.2934074
1: -21.1606407, 32.7483139, -15.0510550, 23.6126156, -44.7732544, 47.7993698
2: -21.0287209, 32.0600052, -14.9348707, 22.9084454, -43.9371643, 46.9948769
3: -26.6253719, 38.4386711, -19.1946678, 27.8182869, -54.4436569, 57.6333389
4: -23.9056454, 36.2302704, -17.2951431, 26.0583858, -49.9640312, 53.5254097

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3496486, upper bound: 46.4022237
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3188378, upper bound: 46.3748583
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3503839, upper bound: 46.4005789
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -18.5107365, 33.2032585, -13.0310068, 23.3925934, -41.9033203, 46.2342644
1: -21.1606407, 32.7483139, -14.9737244, 23.3782749, -44.5389137, 47.7220383
2: -21.0287209, 32.0600052, -14.8151503, 22.7017288, -43.7304420, 46.8751564
3: -26.6253719, 38.4386711, -19.0846157, 27.5269566, -54.1523285, 57.5232849
4: -23.9056454, 36.2302704, -17.0797577, 25.8154736, -49.7211151, 53.3100243

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3477475, upper bound: 46.3949370
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3188755, upper bound: 46.3697266
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3245632, upper bound: 46.3940083
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -20.0294838, 35.6806297, -12.8695917, 23.1978874, -43.2273674, 48.4929962
1: -22.8956375, 35.2464561, -14.8005753, 23.1950188, -46.0906563, 50.0470200
2: -22.7774448, 34.4977379, -14.6490326, 22.5233936, -45.3008385, 49.1467705
3: -28.7865505, 41.4025917, -18.8776188, 27.3054333, -56.0919724, 60.2802086
4: -25.8944168, 39.0450287, -16.9203701, 25.5841904, -51.4786034, 55.9654007

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3494641, upper bound: 46.3943928
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772750, upper bound: 46.4003575
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -20.0294838, 35.6806297, -12.2885456, 22.1907845, -42.2202682, 47.8693314
1: -22.8956375, 35.2464561, -14.1313658, 22.1690311, -45.0646667, 49.3672028
2: -22.7774448, 34.4977379, -13.9499454, 21.5208340, -44.2982788, 48.4476852
3: -28.7865505, 41.4025917, -18.0284634, 26.0830059, -54.8695488, 59.4310493
4: -25.8944168, 39.0450287, -16.0642853, 24.4581051, -50.3525162, 55.1093140

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3492664, upper bound: 46.3925099
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3770773, upper bound: 46.3984747
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -12.8695917, 23.1978874, -45.0535507, 51.3466225
1: -24.9799175, 38.2082138, -14.8005753, 23.1950188, -48.1749344, 53.0087852
2: -24.8422871, 37.3502884, -14.6490326, 22.5233936, -47.3656807, 51.9993172
3: -31.4298725, 44.9164658, -18.8776188, 27.3054333, -58.7353020, 63.7940826
4: -28.2795525, 42.3808212, -16.9203701, 25.5841904, -53.8637390, 59.3011932

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3949972, upper bound: 46.3831175
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986698, upper bound: 46.4015520
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -12.2885456, 22.1907845, -44.0464516, 50.7655792
1: -24.9799175, 38.2082138, -14.1313658, 22.1690311, -47.1489410, 52.3395729
2: -24.8422871, 37.3502884, -13.9499454, 21.5208340, -46.3631134, 51.3002319
3: -31.4298725, 44.9164658, -18.0284634, 26.0830059, -57.5128746, 62.9449272
4: -28.2795525, 42.3808212, -16.0642853, 24.4581051, -52.7376556, 58.4451065

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3866803, upper bound: 46.3993625
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3872477, upper bound: 46.3806663
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3973719, upper bound: 46.3982980
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -12.8695917, 23.1978874, -44.8262749, 50.9881744
1: -24.7276039, 37.7699165, -14.8005753, 23.1950188, -47.9226227, 52.5704880
2: -24.5516720, 36.9445229, -14.6490326, 22.5233936, -47.0750656, 51.5935516
3: -31.0978374, 44.3794708, -18.8776188, 27.3054333, -58.4032555, 63.2570877
4: -27.8886814, 41.8913345, -16.9203701, 25.5841904, -53.4728699, 58.8117065

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3889403, upper bound: 46.3831120
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3926129, upper bound: 46.4015464
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -12.2885456, 22.1907845, -43.8191719, 50.4071350
1: -24.7276039, 37.7699165, -14.1313658, 22.1690311, -46.8966331, 51.9012756
2: -24.5516720, 36.9445229, -13.9499454, 21.5208340, -46.0725060, 50.8944664
3: -31.0978374, 44.3794708, -18.0284634, 26.0830059, -57.1808319, 62.4079323
4: -27.8886814, 41.8913345, -16.0642853, 24.4581051, -52.3467865, 57.9556198

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3889137, upper bound: 46.3993558
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848179, upper bound: 46.3829091
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -18.5107365, 33.2032585, -20.6246643, 36.4598656, -54.8632736, 53.6807060
1: -21.1606407, 32.7483139, -23.5747108, 36.0849686, -57.2149773, 56.2719154
2: -21.0287209, 32.0600052, -23.3903923, 35.2937012, -56.2466888, 55.3535767
3: -26.6253719, 38.4386711, -29.6732693, 42.3714294, -68.9968033, 68.1119232
4: -23.9056454, 36.2302704, -26.5407066, 39.9774590, -63.8677483, 62.7420464

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3412436, upper bound: 46.3191027
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3567496, upper bound: 46.3815470
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -20.0294838, 35.6806297, -20.6246643, 36.4598656, -56.4214554, 56.0946465
1: -22.8956375, 35.2464561, -23.5747108, 36.0849686, -58.9806061, 58.7165031
2: -22.7774448, 34.4977379, -23.3903923, 35.2937012, -57.9861946, 57.7356415
3: -28.7865505, 41.4025917, -29.6732693, 42.3714294, -71.1579742, 71.0309601
4: -25.8944168, 39.0450287, -26.5407066, 39.9774590, -65.8269806, 65.5168304

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3154551, upper bound: 46.3241027
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3545688, upper bound: 46.3579372
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -22.0927143, 38.8323898, -58.8323555, 57.7383881
1: -22.9604015, 35.3986816, -25.2516708, 38.4694748, -61.4298744, 60.6503525
2: -22.8718414, 34.6248589, -25.0594330, 37.6350479, -60.3845406, 59.6097069
3: -28.8905430, 41.6071968, -31.7492695, 45.1954880, -74.0860291, 73.3564453
4: -26.0635490, 39.2244759, -28.4246864, 42.6831512, -68.7111969, 67.6114502

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3556868, upper bound: 46.3981255
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3553415, upper bound: 46.3948238
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -19.9100819, 35.4986420, -22.0927143, 38.8323898, -58.6455994, 57.4813919
1: -22.7595940, 35.0610962, -25.2516708, 38.4694748, -61.2211990, 60.3127556
2: -22.6484280, 34.3139229, -25.0594330, 37.6350479, -60.1641541, 59.3008957
3: -28.6167450, 41.1884537, -31.7492695, 45.1954880, -73.8121948, 72.9376984
4: -25.7594795, 38.8430481, -28.4246864, 42.6831512, -68.4028015, 67.2184982

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3827593, upper bound: 46.4040904
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823995, upper bound: 46.4007887
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.8250885, 38.4325142, -20.6246643, 36.4598656, -58.2536774, 58.9583473
1: -24.9442997, 38.1634560, -23.5747108, 36.0849686, -61.0292664, 61.7105446
2: -24.8085918, 37.3065224, -23.3903923, 35.2937012, -60.0483780, 60.6219254
3: -31.3857899, 44.8639717, -29.6732693, 42.3714294, -73.7572174, 74.5372391
4: -28.2443905, 42.3295364, -26.5407066, 39.9774590, -68.2218475, 68.8702240

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4001282, upper bound: 46.3885531
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986240, upper bound: 46.3596591
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986242, upper bound: 46.3596591
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5980549, 38.0758400, -20.6246643, 36.4598656, -58.0111084, 58.6140099
1: -24.6931915, 37.7268829, -23.5747108, 36.0849686, -60.7781601, 61.2796059
2: -24.5189571, 36.9024200, -23.3903923, 35.2937012, -59.7606163, 60.2252998
3: -31.0551796, 44.3289032, -29.6732693, 42.3714294, -73.4265976, 74.0021515
4: -27.8546429, 41.8416786, -26.5407066, 39.9774590, -67.8320999, 68.3792038

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3953859, upper bound: 46.3885476
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3979686, upper bound: 46.3596591
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3979688, upper bound: 46.3596591
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.3748245, 37.8003426, -22.0927143, 38.8323898, -60.1043015, 59.8326645
1: -24.4454060, 37.4495697, -25.2516708, 38.4694748, -62.8999939, 62.7012329
2: -24.2842541, 36.6351089, -25.0594330, 37.6350479, -61.8010750, 61.6315804
3: -30.7518768, 44.0015488, -31.7492695, 45.1954880, -75.9473495, 75.7508163
4: -27.6241150, 41.5083923, -28.4246864, 42.6831512, -70.2676010, 69.8991165

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4041561, upper bound: 46.4025624
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4028826, upper bound: 46.4025624
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.7154884, 36.6832848, -22.0927143, 38.8323898, -59.4161835, 58.7144623
1: -23.6921539, 36.3185692, -25.2516708, 38.4694748, -62.0983772, 61.5702400
2: -23.5036774, 35.5279503, -25.0594330, 37.6350479, -61.0221634, 60.5169411
3: -29.8063889, 42.6648102, -31.7492695, 45.1954880, -75.0018768, 74.4140778
4: -26.7057209, 40.2674866, -28.4246864, 42.6831512, -69.3554993, 68.6478424

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985877, upper bound: 46.4080346
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3978258, upper bound: 46.4019776
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.2968855, upper bound: 46.3181496
IS_A1_B1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.2968855, upper bound: 46.3181496
IS_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3868266, upper bound: 46.3460774
IS_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3849438, upper bound: 46.3458797
IS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3697266, upper bound: 46.3188755
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3785013, upper bound: 46.3245632
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3783742, upper bound: 46.3683297
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3871489, upper bound: 46.3740174
IS_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3642384, upper bound: 46.3820772
IS_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3716510, upper bound: 46.3923321
IS_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3708134
IS_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3901540
IS_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3915564, upper bound: 46.3972800
IS_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3919843, upper bound: 46.3902990
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3901015, upper bound: 46.3901013
IS_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3902425, upper bound: 46.3429712
IS_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3851667, upper bound: 46.3262504
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3852336, upper bound: 46.3430090
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3940083, upper bound: 46.3486967
IS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3943928, upper bound: 46.3494641
IS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.4003575, upper bound: 46.3772750
IS_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3925099, upper bound: 46.3492664
IS_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3984747, upper bound: 46.3770773
IS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3831175, upper bound: 46.3949972
IS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.4015520, upper bound: 46.3986698
IS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3806663, upper bound: 46.3872477
IS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3982980, upper bound: 46.3428982
IS_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3831120, upper bound: 46.3889403
IS_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.4015466, upper bound: 46.3926129
IS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3993558, upper bound: 46.3889137
IS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3829091, upper bound: 46.3848179
IS_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3188378, upper bound: 46.3748583
IS_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3503839, upper bound: 46.4005789
IS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3188755, upper bound: 46.3697266
IS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3245632, upper bound: 46.3940083
IS_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3494641, upper bound: 46.3943928
IS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3772750, upper bound: 46.4003575
IS_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3492664, upper bound: 46.3925099
IS_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3770773, upper bound: 46.3984747
IS_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3949972, upper bound: 46.3831175
IS_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3986698, upper bound: 46.4015520
IS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3872477, upper bound: 46.3806663
IS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3973719, upper bound: 46.3982980
IS_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3889403, upper bound: 46.3831120
IS_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3926129, upper bound: 46.4015464
IS_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3889137, upper bound: 46.3993558
IS_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3848179, upper bound: 46.3829091
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3412436, upper bound: 46.3191027
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3567496, upper bound: 46.3815470
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3154551, upper bound: 46.3241027
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3545688, upper bound: 46.3579372
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3556868, upper bound: 46.3981255
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3553415, upper bound: 46.3948238
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3827593, upper bound: 46.4040904
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3823995, upper bound: 46.4007887
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3986240, upper bound: 46.3596591
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3986242, upper bound: 46.3596591
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3979686, upper bound: 46.3596591
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3979688, upper bound: 46.3596591
IS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.4041561, upper bound: 46.4025624
IS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.4028826, upper bound: 46.4025624
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3985877, upper bound: 46.4080346
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.52
Output dim: 3, lower bound: -46.3978258, upper bound: 46.4019776

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.7642651, 23.0316162, -11.2742805, 20.7554340, -33.5196991, 34.3058968
1: -14.6807899, 23.0294209, -12.9542065, 20.7355442, -35.4163361, 35.9836273
2: -14.5318680, 22.3620224, -12.9108934, 20.1249809, -34.6568451, 35.2729111
3: -18.7276096, 27.1085167, -16.5860004, 24.4271832, -43.1547928, 43.6945152
4: -16.7909279, 25.3956585, -15.0160170, 22.8133278, -39.6042480, 40.4116745

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3868266, upper bound: 46.3460774
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3868266, upper bound: 46.3460774
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.1827021, 22.0196552, -11.2742805, 20.7554340, -32.9381371, 33.2939339
1: -14.0106850, 22.0007076, -12.9542065, 20.7355442, -34.7462311, 34.9549141
2: -13.8316402, 21.3565445, -12.9108934, 20.1249809, -33.9566116, 34.2674370
3: -17.8771915, 25.8827667, -16.5860004, 24.4271832, -42.3043747, 42.4687653
4: -15.9328604, 24.2661724, -15.0160170, 22.8133278, -38.7461853, 39.2821884

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3844301, upper bound: 46.3445592
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3648488, upper bound: 46.3184752
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2785702, 22.3176193, -9.8099222, 18.3955498, -30.6741123, 32.1275368
1: -14.1255665, 22.2842674, -11.2973719, 18.3045788, -32.4301453, 33.5816383
2: -14.0070620, 21.6412296, -11.2150269, 17.7656765, -31.7727356, 32.8562546
3: -18.0323181, 26.2360859, -14.4795485, 21.5189934, -39.5513000, 40.7156258
4: -16.2166100, 24.5360146, -13.0337639, 20.0603638, -36.2769737, 37.5697746

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3653190, upper bound: 46.3145877
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3514184, upper bound: 46.3130234
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3514184, upper bound: 46.3188755
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.4544268, 24.1434498, -9.8099222, 18.3955498, -31.8499737, 33.9533615
1: -15.4540081, 24.0967407, -11.2973719, 18.3045788, -33.7585869, 35.3941116
2: -15.2934589, 23.4114799, -11.2150269, 17.7656765, -33.0591354, 34.6265068
3: -19.6883392, 28.3864269, -14.4795485, 21.5189934, -41.2073250, 42.8659744
4: -17.6013470, 26.5991116, -13.0337639, 20.0603638, -37.6617126, 39.6328697

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3757468, upper bound: 46.3220012
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3630173, upper bound: 46.3222108
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3630173, upper bound: 46.3245632
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -12.3053322, 22.3548565, -11.1346703, 20.5469704, -32.8522987, 33.4895134
1: -14.1560287, 22.3223190, -12.8063316, 20.4618702, -34.6178970, 35.1286507
2: -14.0357513, 21.6778736, -12.7347527, 19.8716698, -33.9074173, 34.4126244
3: -18.0700207, 26.2806225, -16.3541832, 24.0840511, -42.1540604, 42.6348038
4: -16.2465935, 24.5797844, -14.7644615, 22.4989681, -38.7455597, 39.3442459

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3717666, upper bound: 46.3571577
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3740222, upper bound: 46.3646422
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -13.4835348, 24.1837368, -11.1346703, 20.5469704, -34.0304985, 35.3183937
1: -15.4869547, 24.1380043, -12.8063316, 20.4618702, -35.9488258, 36.9443359
2: -15.3245106, 23.4516239, -12.7347527, 19.8716698, -35.1961708, 36.1863708
3: -19.7292652, 28.4352989, -16.3541832, 24.0840511, -43.8133163, 44.7894783
4: -17.6341152, 26.6464577, -14.7644615, 22.4989681, -40.1330833, 41.4109116

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3821945, upper bound: 46.3645711
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3844500, upper bound: 46.3720556
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -11.9084167, 21.7838936, -33.6642151, 33.4091225
1: -13.6646070, 21.4798603, -13.6919727, 21.8264008, -35.4910088, 35.1718330
2: -13.4940720, 20.8352280, -13.6413050, 21.1797218, -34.6737900, 34.4765244
3: -17.4712029, 25.2716560, -17.5283012, 25.7199497, -43.1911469, 42.7999535
4: -15.5687332, 23.6612968, -15.9078255, 24.0077763, -39.5765076, 39.5691147

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3625199, upper bound: 46.3808564
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3541289, upper bound: 46.3783690
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3541289, upper bound: 46.3820772
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -12.9794855, 23.4814758, -35.3617935, 34.4801903
1: -13.6646070, 21.4798603, -14.9135637, 23.5045586, -37.1691589, 36.3934250
2: -13.4940720, 20.8352280, -14.8262920, 22.8127975, -36.3068695, 35.6615181
3: -17.4712029, 25.2716560, -19.0450211, 27.7114143, -45.1826172, 44.3166771
4: -15.5687332, 23.6612968, -17.1786613, 25.9186611, -41.4873962, 40.8399582

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3707403, upper bound: 46.3911498
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3615416, upper bound: 46.3886854
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3716510, upper bound: 46.3923321
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3705767, upper bound: 46.3894313
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -11.3531218, 20.6934528, -32.5737724, 32.8538284
1: -13.6646070, 21.4798603, -13.0675001, 20.6743774, -34.3389816, 34.5473595
2: -13.4940720, 20.8352280, -12.9131918, 20.0512848, -33.5453529, 33.7484207
3: -17.4712029, 25.2716560, -16.7260056, 24.3298912, -41.8010941, 41.9976540
4: -15.5687332, 23.6612968, -14.9432716, 22.7504520, -38.3191795, 38.6045685

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3627291, upper bound: 46.3560098
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3661196, upper bound: 46.3661193
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -12.6439228, 22.7910843, -34.6714020, 34.1446304
1: -13.6646070, 21.4798603, -14.5368881, 22.7865276, -36.4511223, 36.0167465
2: -13.4940720, 20.8352280, -14.3889275, 22.1210289, -35.6151009, 35.2241554
3: -17.4712029, 25.2716560, -18.5426521, 26.8304729, -44.3016739, 43.8143082
4: -15.5687332, 23.6612968, -16.6194782, 25.1390991, -40.7078323, 40.2807732

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3627291, upper bound: 46.3768822
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3661196, upper bound: 46.3856570
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -12.3138046, 22.3993435, -35.5915565, 35.9429550
1: -15.1584854, 23.6275005, -14.1712494, 22.4655819, -37.6240692, 37.7987518
2: -14.9872761, 22.9454288, -14.0879164, 21.7890587, -36.7763367, 37.0333443
3: -19.3160419, 27.8168468, -18.1163712, 26.4603176, -45.7763596, 45.9332123
4: -17.2645493, 26.0874233, -16.3892231, 24.7468948, -42.0114403, 42.4766464

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3751097, upper bound: 46.3931842
time: 1.25 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3744231, upper bound: 46.3917760
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -11.8822479, 21.6655045, -34.8577194, 35.5113945
1: -15.1584854, 23.6275005, -13.6782455, 21.7105942, -36.8690796, 37.3057480
2: -14.9872761, 22.9454288, -13.5535717, 21.0439568, -36.0312347, 36.4990005
3: -19.3160419, 27.8168468, -17.4800777, 25.5661774, -44.8822174, 45.2969246
4: -17.2645493, 26.0874233, -15.7685585, 23.9080009, -41.1725426, 41.8559799

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.7642651, 23.0316162, -12.7501745, 22.9591522, -35.7234154, 35.7817802
1: -14.6807899, 23.0294209, -14.6577673, 22.9542313, -37.6350174, 37.6871834
2: -14.5318680, 22.3620224, -14.5072327, 22.2842579, -36.8161240, 36.8692513
3: -18.7276096, 27.1085167, -18.6941471, 27.0294781, -45.7570877, 45.8026543
4: -16.7909279, 25.3956585, -16.7502651, 25.3299522, -42.1208763, 42.1459236

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3919843, upper bound: 46.3902990
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3919843, upper bound: 46.3902990
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1827021, 22.0196552, -12.7501745, 22.9591522, -35.1418495, 34.7698212
1: -14.0106850, 22.0007076, -14.6577673, 22.9542313, -36.9649162, 36.6584663
2: -13.8316402, 21.3565445, -14.5072327, 22.2842579, -36.1158981, 35.8637772
3: -17.8771915, 25.8827667, -18.6941471, 27.0294781, -44.9066696, 44.5769119
4: -15.9328604, 24.2661724, -16.7502651, 25.3299522, -41.2628136, 41.0164375

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3901015, upper bound: 46.3901013
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3901015, upper bound: 46.3901013
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -12.1836224, 22.2152824, -18.5107365, 33.2032585, -45.3868713, 40.7260132
1: -14.0123215, 22.2466698, -21.1606407, 32.7483139, -46.7606354, 43.4073105
2: -13.9481668, 21.5913982, -21.0287209, 32.0600052, -46.0081635, 42.6201172
3: -17.9143009, 26.2163086, -26.6253719, 38.4386711, -56.3529739, 52.8416824
4: -16.2357998, 24.4893208, -23.9056454, 36.2302704, -52.4660721, 48.3949623

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3733578, upper bound: 46.3156819
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
time: 0.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 3, lower bound: -46.4095348, upper bound: 46.4032405

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -15.6748066, 27.2767467, -40.5789375, 39.4772720
1: -15.2836475, 23.8005772, -17.9646091, 27.3548412, -42.6384850, 41.7651787
2: -15.1096706, 23.1136723, -17.7045784, 26.5651569, -41.6748276, 40.8182526
3: -19.4732094, 28.0227852, -22.8100491, 32.1917152, -51.6649170, 50.8328323
4: -17.4002285, 26.2846336, -20.1704769, 30.4162006, -47.8164215, 46.4551048

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
time: 0.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
time: 0.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -15.6405516, 27.2283001, -49.4413528, 54.6583061
1: -25.3891716, 38.6557007, -17.9264469, 27.3087788, -52.6979446, 56.5821457
2: -25.1929913, 37.8162651, -17.6665192, 26.5205688, -51.7135582, 55.4827843
3: -31.9208298, 45.4162788, -22.7647934, 32.1371918, -64.0580215, 68.1810760
4: -28.5711136, 42.8956490, -20.1301193, 30.3629417, -58.9340553, 63.0257568

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
time: 0.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113062
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 3, lower bound: -46.4014513, upper bound: 46.4014513
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4095348
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 3, lower bound: -46.4032405, upper bound: 46.4113062

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.3021946, 23.8024693, -37.1046600, 37.1046600
1: -15.2836475, 23.8005772, -15.2836475, 23.8005772, -39.0842171, 39.0842209
2: -15.1096706, 23.1136723, -15.1096706, 23.1136723, -38.2233429, 38.2233429
3: -19.4732094, 28.0227852, -19.4732094, 28.0227852, -47.4959869, 47.4959869
4: -17.4002285, 26.2846336, -17.4002285, 26.2846336, -43.6848602, 43.6848602

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3784407
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
time: 0.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -22.2130566, 39.0177612, -52.3199539, 46.0155220
1: -15.2836475, 23.8005772, -25.3891716, 38.6557007, -53.9393463, 49.1897392
2: -15.1096706, 23.1136723, -25.1929913, 37.8162651, -52.9259338, 48.3066559
3: -19.4732094, 28.0227852, -31.9208298, 45.4162788, -64.8894882, 59.9436111
4: -17.4002285, 26.2846336, -28.5711136, 42.8956490, -60.2958755, 54.8557472

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3814746
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4025283
time: 0.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -13.3021946, 23.8024693, -46.0155220, 52.3199539
1: -25.3891716, 38.6557007, -15.2836475, 23.8005772, -49.1897392, 53.9393463
2: -25.1929913, 37.8162651, -15.1096706, 23.1136723, -48.3066559, 52.9259338
3: -31.9208298, 45.4162788, -19.4732094, 28.0227852, -59.9436111, 64.8894806
4: -28.5711136, 42.8956490, -17.4002285, 26.2846336, -54.8557472, 60.2958755

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4025283, upper bound: 46.4087713
time: 1.01 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -22.2130566, 39.0177612, -61.2015877, 61.2015839
1: -25.3891716, 38.6557007, -25.3891716, 38.6557007, -64.0448761, 64.0448761
2: -25.1929913, 37.8162651, -25.1929913, 37.8162651, -62.9735107, 62.9735184
3: -31.9208298, 45.4162788, -31.9208298, 45.4162788, -77.3370972, 77.3370972
4: -28.5711136, 42.8956490, -28.5711136, 42.8956490, -71.4667664, 71.4667664

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784407, upper bound: 46.3966428
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4025283, upper bound: 46.4105995
time: 1.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.57 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3784407
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4006822
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.3966428, upper bound: 46.3814746
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.4006824, upper bound: 46.4025283
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.3814746, upper bound: 46.4071147
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.4025283, upper bound: 46.4087713
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.3784407, upper bound: 46.3966428
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -46.4025283, upper bound: 46.4105995

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -11.3869448, 20.9255638, -34.2277527, 35.1894150
1: -15.2836475, 23.8005772, -13.0953932, 20.8397293, -36.1233749, 36.8959579
2: -15.1096706, 23.1136723, -13.0110359, 20.2390823, -35.3487549, 36.1247101
3: -19.4732094, 28.0227852, -16.7173176, 24.5306568, -44.0038528, 44.7401009
4: -17.4002285, 26.2846336, -15.0642700, 22.9335842, -40.3338127, 41.3489037

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.0015049, 23.3439465, -36.6461411, 36.8039742
1: -15.2836475, 23.8005772, -14.9456024, 23.3515148, -38.6351547, 38.7461777
2: -15.1096706, 23.1136723, -14.7805185, 22.6720467, -37.7817154, 37.8941917
3: -19.4732094, 28.0227852, -19.0547180, 27.4934063, -46.9666100, 47.0775032
4: -17.4002285, 26.2846336, -17.0490189, 25.7683182, -43.1685410, 43.3336525

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3939982
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -20.1475525, 35.8621483, -49.1643410, 43.9500198
1: -15.2836475, 23.8005772, -23.0306740, 35.4290886, -50.7127342, 46.8312416
2: -15.1096706, 23.1136723, -22.9083958, 34.6756935, -49.7853622, 46.0220680
3: -19.4732094, 28.0227852, -28.9547653, 41.6189651, -61.0921669, 56.9775505
4: -17.4002285, 26.2846336, -26.0378189, 39.2533302, -56.6535568, 52.3224525

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3506872
time: 0.98 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3797874
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -21.9027481, 38.5440521, -51.8462448, 45.7052155
1: -15.2836475, 23.8005772, -25.0402546, 38.1956673, -53.4793167, 48.8408279
2: -15.1096706, 23.1136723, -24.8525200, 37.3617020, -52.4713745, 47.9661942
3: -19.4732094, 28.0227852, -31.4891014, 44.8785400, -64.3517456, 59.5118866
4: -17.4002285, 26.2846336, -28.2129040, 42.3660202, -59.7662506, 54.4975357

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4037566, upper bound: 46.4008411
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3978337
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -13.3021946, 23.8024693, -43.9500198, 49.1643410
1: -23.0306740, 35.4290886, -15.2836475, 23.8005772, -46.8312416, 50.7127342
2: -22.9083958, 34.6756935, -15.1096706, 23.1136723, -46.0220680, 49.7853622
3: -28.9547653, 41.6189651, -19.4732094, 28.0227852, -56.9775505, 61.0921669
4: -26.0378189, 39.2533302, -17.4002285, 26.2846336, -52.3224525, 56.6535568

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3470623, upper bound: 46.3845465
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3470623, upper bound: 46.3934066
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -13.3021946, 23.8024693, -45.7052155, 51.8462448
1: -25.0402546, 38.1956673, -15.2836475, 23.8005772, -48.8408241, 53.4793167
2: -24.8525200, 37.3617020, -15.1096706, 23.1136723, -47.9661942, 52.4713745
3: -31.4891014, 44.8785400, -19.4732094, 28.0227852, -59.5118866, 64.3517456
4: -28.2129040, 42.3660202, -17.4002285, 26.2846336, -54.4975357, 59.7662506

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4008411, upper bound: 46.4037566
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3978337, upper bound: 46.4037510
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -22.2130566, 39.0177612, -59.1370125, 57.9924889
1: -23.0306740, 35.4290886, -25.3891716, 38.6557007, -61.6863747, 60.8182602
2: -22.9083958, 34.6756935, -25.1929913, 37.8162651, -60.6736412, 59.8156548
3: -28.9547653, 41.6189651, -31.9208298, 45.4162788, -74.3710175, 73.5397873
4: -26.0378189, 39.2533302, -28.5711136, 42.8956490, -68.9334717, 67.8180923

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3566082, upper bound: 46.3952419
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3470623, upper bound: 46.3916459
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -22.2130566, 39.0177612, -60.8921165, 60.7265244
1: -25.0402546, 38.1956673, -25.3891716, 38.6557007, -63.6959534, 63.5848389
2: -24.8525200, 37.3617020, -25.1929913, 37.8162651, -62.6320419, 62.5175209
3: -31.4891014, 44.8785400, -31.9208298, 45.4162788, -76.9053802, 76.7993622
4: -28.2129040, 42.3660202, -28.5711136, 42.8956490, -71.1085510, 70.9371338

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4043732, upper bound: 46.3895545
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966428, upper bound: 46.4006822
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.98 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3939982
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3506872
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3797874
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.4037566, upper bound: 46.4008411
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3978337
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3470623, upper bound: 46.3845465
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3470623, upper bound: 46.3934066
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.4008411, upper bound: 46.4037566
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3978337, upper bound: 46.4037510
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3566082, upper bound: 46.3952419
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3470623, upper bound: 46.3916459
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.4043732, upper bound: 46.3895545
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3966428, upper bound: 46.4006822

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -13.2286949, 23.6841087, -11.2742805, 20.7554340, -33.9841309, 34.9583893
1: -15.2004843, 23.6871033, -12.9542065, 20.7355442, -35.9360237, 36.6413116
2: -15.0275564, 23.0018730, -12.9108934, 20.1249809, -35.1525383, 35.9127617
3: -19.3699303, 27.8896084, -16.5860004, 24.4271832, -43.7971077, 44.4756088
4: -17.3119068, 26.1560364, -15.0160170, 22.8133278, -40.1252327, 41.1720543

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -11.2217398, 20.6840553, -33.9862404, 35.0242081
1: -15.2836475, 23.8005772, -12.9060535, 20.5976448, -35.8812904, 36.7066231
2: -15.1096706, 23.1136723, -12.8317413, 20.0037327, -35.1134033, 35.9454117
3: -19.4732094, 28.0227852, -16.4788857, 24.2461224, -43.7193298, 44.5016708
4: -17.4002285, 26.2846336, -14.8719177, 22.6551094, -40.0553360, 41.1565514

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.2286949, 23.6841087, -12.8240356, 23.1475544, -36.3762512, 36.5081367
1: -15.2004843, 23.6871033, -14.7506380, 23.2159920, -38.4164772, 38.4377403
2: -15.0275564, 23.0018730, -14.6458588, 22.5191956, -37.5467529, 37.6477242
3: -19.3699303, 27.8896084, -18.8241386, 27.3499374, -46.7198524, 46.7137451
4: -17.3119068, 26.1560364, -16.9858208, 25.6044750, -42.9163818, 43.1418571

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3749012, upper bound: 46.3926650
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -12.7501745, 22.9591522, -36.2613373, 36.5526390
1: -15.2836475, 23.8005772, -14.6577673, 22.9542313, -38.2378769, 38.4583321
2: -15.1096706, 23.1136723, -14.5072327, 22.2842579, -37.3939285, 37.6209030
3: -19.4732094, 28.0227852, -18.6941471, 27.0294781, -46.5026817, 46.7169342
4: -17.4002285, 26.2846336, -16.7502651, 25.3299522, -42.7301788, 43.0348969

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3899862, upper bound: 46.3741746
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3262267
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -13.2286949, 23.6841087, -20.0963440, 35.7420425, -48.9707375, 43.7804527
1: -15.2004843, 23.6871033, -22.9604015, 35.3986816, -50.5991631, 46.6475067
2: -15.0275564, 23.0018730, -22.8718414, 34.6248589, -49.6524162, 45.8737106
3: -19.3699303, 27.8896084, -28.8905430, 41.6071968, -60.9771156, 56.7801514
4: -17.3119068, 26.1560364, -26.0635490, 39.2244759, -56.5363770, 52.2195854

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3506872
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -19.9100819, 35.4986420, -48.8008347, 43.7125473
1: -15.2836475, 23.8005772, -22.7595940, 35.0610962, -50.3447418, 46.5601692
2: -15.1096706, 23.1136723, -22.6484280, 34.3139229, -49.4235916, 45.7621002
3: -19.4732094, 28.0227852, -28.6167450, 41.1884537, -60.6616631, 56.6395264
4: -17.4002285, 26.2846336, -25.7594795, 38.8430481, -56.2432747, 52.0441132

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3979147, upper bound: 46.3496843
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4020900, upper bound: 46.3797874
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -13.2286949, 23.6841087, -21.8556671, 38.4770317, -51.7057266, 45.5397720
1: -15.2004843, 23.6871033, -24.9799175, 38.2082138, -53.4086990, 48.6670227
2: -15.0275564, 23.0018730, -24.8422871, 37.3502884, -52.3778419, 47.8441544
3: -19.3699303, 27.8896084, -31.4298725, 44.9164658, -64.2863922, 59.3194809
4: -17.3119068, 26.1560364, -28.2795525, 42.3808212, -59.6927261, 54.4355888

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015443, upper bound: 46.3986698
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996692, upper bound: 46.3984722
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -21.6283913, 38.1185913, -51.4207802, 45.4308548
1: -15.2836475, 23.8005772, -24.7276039, 37.7699165, -53.0535622, 48.5281715
2: -15.1096706, 23.1136723, -24.5516720, 36.9445229, -52.0541916, 47.6653442
3: -19.4732094, 28.0227852, -31.0978374, 44.3794708, -63.8526726, 59.1206169
4: -17.4002285, 26.2846336, -27.8886814, 41.8913345, -59.2915611, 54.1733170

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015280, upper bound: 46.3926129
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996638, upper bound: 46.3924153
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -13.2286949, 23.6841087, -43.7804527, 48.9707375
1: -22.9604015, 35.3986816, -15.2004843, 23.6871033, -46.6475067, 50.5991669
2: -22.8718414, 34.6248589, -15.0275564, 23.0018730, -45.8737106, 49.6524162
3: -28.8905430, 41.6071968, -19.3699303, 27.8896084, -56.7801514, 60.9771156
4: -26.0635490, 39.2244759, -17.3119068, 26.1560364, -52.2195854, 56.5363808

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3506872, upper bound: 46.3934066
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3506872, upper bound: 46.3934066
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -19.9100819, 35.4986420, -13.3021946, 23.8024693, -43.7125473, 48.8008347
1: -22.7595940, 35.0610962, -15.2836475, 23.8005772, -46.5601692, 50.3447342
2: -22.6484280, 34.3139229, -15.1096706, 23.1136723, -45.7621002, 49.4235916
3: -28.6167450, 41.1884537, -19.4732094, 28.0227852, -56.6395264, 60.6616592
4: -25.7594795, 38.8430481, -17.4002285, 26.2846336, -52.0441132, 56.2432747

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3496844, upper bound: 46.3979149
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020899
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -13.2286949, 23.6841087, -45.5397758, 51.7057266
1: -24.9799175, 38.2082138, -15.2004843, 23.6871033, -48.6670227, 53.4086990
2: -24.8422871, 37.3502884, -15.0275564, 23.0018730, -47.8441544, 52.3778419
3: -31.4298725, 44.9164658, -19.3699303, 27.8896084, -59.3194809, 64.2863922
4: -28.2795525, 42.3808212, -17.3119068, 26.1560364, -54.4355888, 59.6927261

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986698, upper bound: 46.4015443
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3984722, upper bound: 46.3996692
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -13.3021946, 23.8024693, -45.4308586, 51.4207802
1: -24.7276039, 37.7699165, -15.2836475, 23.8005772, -48.5281715, 53.0535622
2: -24.5516720, 36.9445229, -15.1096706, 23.1136723, -47.6653442, 52.0541916
3: -31.0978374, 44.3794708, -19.4732094, 28.0227852, -59.1206169, 63.8526764
4: -27.8886814, 41.8913345, -17.4002285, 26.2846336, -54.1733170, 59.2915611

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3748822, upper bound: 46.3919843
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3924153, upper bound: 46.3996636
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -22.1349335, 38.8901787, -58.9443855, 57.8043518
1: -22.9604015, 35.3986816, -25.3007240, 38.5336723, -61.4940681, 60.6994057
2: -22.8718414, 34.6248589, -25.1061096, 37.6958046, -60.5016670, 59.6777916
3: -28.8905430, 41.6071968, -31.8115883, 45.2737465, -74.1642914, 73.4187851
4: -26.0635490, 39.2244759, -28.4783401, 42.7584648, -68.8220062, 67.7028122

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3566082, upper bound: 46.3952419
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3566082, upper bound: 46.3952419
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -19.9100819, 35.4986420, -22.2130566, 39.0177612, -58.8958626, 57.6247444
1: -22.7595940, 35.0610962, -25.3891716, 38.6557007, -61.4152946, 60.4502602
2: -22.6484280, 34.3139229, -25.1929913, 37.8162651, -60.4099541, 59.4551849
3: -28.6167450, 41.1884537, -31.9208298, 45.4162788, -74.0329819, 73.1092682
4: -25.7594795, 38.8430481, -28.5711136, 42.8956490, -68.6551285, 67.4047241

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3593334, upper bound: 46.4019957
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3857363, upper bound: 46.4059534
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -20.1475525, 35.8621483, -57.6830177, 58.6619530
1: -25.0402546, 38.1956673, -23.0306740, 35.4290886, -60.4693413, 61.2263412
2: -24.8525200, 37.3617020, -22.9083958, 34.6756935, -59.4741821, 60.2176437
3: -31.4891014, 44.8785400, -28.9547653, 41.6189651, -73.1080627, 73.8332977
4: -28.2129040, 42.3660202, -26.0378189, 39.2533302, -67.4620667, 68.4038315

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4043731, upper bound: 46.3895546
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3449396, upper bound: 46.3895490
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -21.9027481, 38.5440521, -60.4170532, 60.4170532
1: -25.0402546, 38.1956673, -25.0402546, 38.1956673, -63.2359238, 63.2359200
2: -24.8525200, 37.3617020, -24.8525200, 37.3617020, -62.1760521, 62.1760521
3: -31.4891014, 44.8785400, -31.4891014, 44.8785400, -76.3676453, 76.3676453
4: -28.2129040, 42.3660202, -28.2129040, 42.3660202, -70.5789185, 70.5789185

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3921170, upper bound: 46.3596710
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4029916, upper bound: 46.4076066
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.81 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
IS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3916459, upper bound: 46.3767535
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3749012, upper bound: 46.3926650
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
IS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3899862, upper bound: 46.3741746
IS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3939984, upper bound: 46.3262267
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3845465, upper bound: 46.3470623
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3506872
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3979147, upper bound: 46.3496843
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.4020900, upper bound: 46.3797874
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.4015443, upper bound: 46.3986698
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3996692, upper bound: 46.3984722
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.4015280, upper bound: 46.3926129
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3996638, upper bound: 46.3924153
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3506872, upper bound: 46.3934066
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3506872, upper bound: 46.3934066
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3496844, upper bound: 46.3979149
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020899
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3986698, upper bound: 46.4015443
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3984722, upper bound: 46.3996692
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3748822, upper bound: 46.3919843
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3924153, upper bound: 46.3996636
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3566082, upper bound: 46.3952419
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3566082, upper bound: 46.3952419
IS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3593334, upper bound: 46.4019957
IS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3857363, upper bound: 46.4059534
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.4043731, upper bound: 46.3895546
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3449396, upper bound: 46.3895490
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.3921170, upper bound: 46.3596710
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 3, lower bound: -46.4029916, upper bound: 46.4076066

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1184626, 23.5935650, -11.2742805, 20.7554340, -33.8738976, 34.8678436
1: -15.0832462, 23.6534405, -12.9542065, 20.7355442, -35.8187904, 36.6076469
2: -14.9651690, 22.9481583, -12.9108934, 20.1249809, -35.0901451, 35.8590508
3: -19.2343731, 27.8664894, -16.5860004, 24.4271832, -43.6615562, 44.4524918
4: -17.3274384, 26.1050282, -15.0160170, 22.8133278, -40.1407623, 41.1210365

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3418199
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3470623
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.0589924, 23.4311523, -11.2742805, 20.7554340, -33.8144264, 34.7054329
1: -15.0052748, 23.4174843, -12.9542065, 20.7355442, -35.7408180, 36.3716888
2: -14.8450346, 22.7399940, -12.9108934, 20.1249809, -34.9700165, 35.6508865
3: -19.1238232, 27.5731144, -16.5860004, 24.4271832, -43.5510063, 44.1591148
4: -17.1109009, 25.8610153, -15.0160170, 22.8133278, -39.9242249, 40.8770332

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3720594, upper bound: 46.3394178
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3818160, upper bound: 46.3470623
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1184626, 23.5935650, -11.2217398, 20.6840553, -33.8025169, 34.8153038
1: -15.0832462, 23.6534405, -12.9060535, 20.5976448, -35.6808891, 36.5594902
2: -14.9651690, 22.9481583, -12.8317413, 20.0037327, -34.9688988, 35.7798996
3: -19.2343731, 27.8664894, -16.4788857, 24.2461224, -43.4804955, 44.3453751
4: -17.3274384, 26.1050282, -14.8719177, 22.6551094, -39.9825478, 40.9769440

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3710746
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3710746
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -13.0589924, 23.4311523, -11.2217398, 20.6840553, -33.7430420, 34.6528931
1: -15.0052748, 23.4174843, -12.9060535, 20.5976448, -35.6029167, 36.3235359
2: -14.8450346, 22.7399940, -12.8317413, 20.0037327, -34.8487663, 35.5717354
3: -19.1238232, 27.5731144, -16.4788857, 24.2461224, -43.3699455, 44.0520020
4: -17.1109009, 25.8610153, -14.8719177, 22.6551094, -39.7660103, 40.7329330

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3739221
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3746640
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -11.8087902, 21.3844547, -12.4869890, 22.6631680, -34.4719582, 33.8714409
1: -13.5835524, 21.3685608, -14.3620119, 22.7225075, -36.3060608, 35.7305679
2: -13.4141731, 20.7254810, -14.2810307, 22.0392475, -35.4534187, 35.0065117
3: -17.3705254, 25.1409931, -18.3535480, 26.7719860, -44.1425095, 43.4945412
4: -15.4819202, 23.5353012, -16.5992470, 25.0412960, -40.5232162, 40.1345482

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.2572491, upper bound: 46.3276246
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3513339, upper bound: 46.3693564
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3746895, upper bound: 46.3926650
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1193733, 23.5113163, -12.8240356, 23.1475544, -36.2669258, 36.3353462
1: -15.0760221, 23.5147686, -14.7506380, 23.2159920, -38.2920151, 38.2654076
2: -14.9058371, 22.8341923, -14.6458588, 22.5191956, -37.4250336, 37.4800491
3: -19.2136593, 27.6845913, -18.8241386, 27.3499374, -46.5635948, 46.5087280
4: -17.1769733, 25.9597168, -16.9858208, 25.6044750, -42.7814484, 42.9455376

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -12.9556446, 23.3202438, -11.3531218, 20.6934528, -33.6490974, 34.6733665
1: -14.8924294, 23.3074780, -13.0675001, 20.6743774, -35.5668068, 36.3749733
2: -14.7394896, 22.6328850, -12.9131918, 20.0512848, -34.7907715, 35.5460739
3: -18.9874325, 27.4399891, -16.7260056, 24.3298912, -43.3173218, 44.1659889
4: -17.0107269, 25.7165833, -14.9432716, 22.7504520, -39.7611732, 40.6598511

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3768760, upper bound: 46.3634878
time: 1.07 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3855924, upper bound: 46.3695266
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -12.6439228, 22.7910843, -36.0932693, 36.4463882
1: -15.2836475, 23.8005772, -14.5368881, 22.7865276, -38.0701637, 38.3374596
2: -15.1096706, 23.1136723, -14.3889275, 22.1210289, -37.2306976, 37.5026016
3: -19.4732094, 28.0227852, -18.5426521, 26.8304729, -46.3036728, 46.5654373
4: -17.4002285, 26.2846336, -16.6194782, 25.1390991, -42.5393257, 42.9041138

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3741748, upper bound: 46.3899860
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3741748, upper bound: 46.3939982
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1184626, 23.5935650, -20.0963440, 35.7420425, -48.8605042, 43.6899109
1: -15.0832462, 23.6534405, -22.9604015, 35.3986816, -50.4819260, 46.6138382
2: -14.9651690, 22.9481583, -22.8718414, 34.6248589, -49.5900192, 45.8199921
3: -19.2343731, 27.8664894, -28.8905430, 41.6071968, -60.8415680, 56.7570305
4: -17.3274384, 26.1050282, -26.0635490, 39.2244759, -56.5519028, 52.1685715

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3912741, upper bound: 46.3490166
time: 1.16 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3917556, upper bound: 46.3491975
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.0589924, 23.4311523, -20.0963440, 35.7420425, -48.8010330, 43.5274963
1: -15.0052748, 23.4174843, -22.9604015, 35.3986816, -50.4039497, 46.3778763
2: -14.8450346, 22.7399940, -22.8718414, 34.6248589, -49.4698906, 45.6118355
3: -19.1238232, 27.5731144, -28.8905430, 41.6071968, -60.7310181, 56.4636536
4: -17.1109009, 25.8610153, -26.0635490, 39.2244759, -56.3353729, 51.9245644

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3912741, upper bound: 46.3490166
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3917556, upper bound: 46.3491975
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -12.9556446, 23.3202438, -18.2608814, 32.8123665, -45.7680130, 41.5811234
1: -14.8924294, 23.3074780, -20.8757095, 32.3502274, -47.2426567, 44.1831894
2: -14.7394896, 22.6328850, -20.7539730, 31.6704140, -46.4099045, 43.3868561
3: -18.9874325, 27.4399891, -26.2691593, 37.9729080, -56.9603424, 53.7091484
4: -17.0107269, 25.7165833, -23.6085434, 35.7925186, -52.8032379, 49.3251266

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3973723, upper bound: 46.3496303
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3979147, upper bound: 46.3496843
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -19.7945900, 35.3213921, -48.5638847, 43.5970573
1: -15.2836475, 23.8005772, -22.6274872, 34.8829269, -50.1665726, 46.4280548
2: -15.1096706, 23.1136723, -22.5204334, 34.1403465, -49.2500153, 45.6341057
3: -19.4732094, 28.0227852, -28.4521999, 40.9773903, -60.4505920, 56.4749832
4: -17.4002285, 26.2846336, -25.6193924, 38.6395912, -56.0398178, 51.9040260

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3797874
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3769216
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -12.7971506, 23.0809841, -21.8556671, 38.4770317, -51.2741814, 44.9366493
1: -14.7185745, 23.0831375, -24.9799175, 38.2082138, -52.9267883, 48.0630569
2: -14.5681829, 22.4129333, -24.8422871, 37.3502884, -51.9184723, 47.2552185
3: -18.7758389, 27.1741924, -31.4298725, 44.9164658, -63.6922913, 58.6040649
4: -16.8335400, 25.4575272, -28.2795525, 42.3808212, -59.2143555, 53.7370796

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4012453, upper bound: 46.3868779
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4012453, upper bound: 46.3868779
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.2123003, 22.0635204, -21.7094116, 38.2469978, -50.4592896, 43.7729340
1: -14.0449352, 22.0455666, -24.8219280, 37.9755707, -52.0205002, 46.8674927
2: -13.8646946, 21.3987408, -24.6786175, 37.1231232, -50.9878159, 46.0773544
3: -17.9215183, 25.9386024, -31.2300510, 44.6412430, -62.5627594, 57.1686478
4: -15.9714890, 24.3215561, -28.0933685, 42.1199799, -58.0914688, 52.4149246

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993626, upper bound: 46.3866803
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3829158, upper bound: 46.3825844
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8695917, 23.1978874, -21.6283913, 38.1185913, -50.9881783, 44.8262711
1: -14.8005753, 23.1950188, -24.7276039, 37.7699165, -52.5704880, 47.9226227
2: -14.6490326, 22.5233936, -24.5516720, 36.9445229, -51.5935555, 47.0750656
3: -18.8776188, 27.3054333, -31.0978374, 44.3794708, -63.2570877, 58.4032555
4: -16.9203701, 25.5841904, -27.8886814, 41.8913345, -58.8117027, 53.4728699

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4012386, upper bound: 46.3891114
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4012386, upper bound: 46.3891114
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.2885456, 22.1907845, -21.4741402, 37.8610115, -50.1495590, 43.6649246
1: -14.1313658, 22.1690311, -24.5510635, 37.5095062, -51.6408653, 46.7200890
2: -13.9499454, 21.5208340, -24.3705120, 36.6906548, -50.6405983, 45.8913383
3: -18.0284634, 26.0830059, -30.8752117, 44.0720291, -62.1004791, 56.9582062
4: -16.0642853, 24.4581051, -27.6808262, 41.6019783, -57.6662636, 52.1389236

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993558, upper bound: 46.3889137
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3829091, upper bound: 46.3848179
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -13.1184626, 23.5935650, -43.6899109, 48.8605042
1: -22.9604015, 35.3986816, -15.0832462, 23.6534405, -46.6138382, 50.4819260
2: -22.8718414, 34.6248589, -14.9651690, 22.9481583, -45.8199921, 49.5900192
3: -28.8905430, 41.6071968, -19.2343731, 27.8664894, -56.7570305, 60.8415680
4: -26.0635490, 39.2244759, -17.3274384, 26.1050282, -52.1685715, 56.5519066

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3490165, upper bound: 46.3912741
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3491975, upper bound: 46.3917556
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -13.0589924, 23.4311523, -43.5274963, 48.8010330
1: -22.9604015, 35.3986816, -15.0052748, 23.4174843, -46.3778801, 50.4039536
2: -22.8718414, 34.6248589, -14.8450346, 22.7399940, -45.6118355, 49.4698906
3: -28.8905430, 41.6071968, -19.1238232, 27.5731144, -56.4636536, 60.7310181
4: -26.0635490, 39.2244759, -17.1109009, 25.8610153, -51.9245644, 56.3353691

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3490165, upper bound: 46.3912741
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3491975, upper bound: 46.3917556
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -18.2608814, 32.8123665, -12.9556446, 23.3202438, -41.5811234, 45.7680130
1: -20.8757095, 32.3502274, -14.8924294, 23.3074780, -44.1831894, 47.2426567
2: -20.7539730, 31.6704140, -14.7394896, 22.6328850, -43.3868561, 46.4099045
3: -26.2691593, 37.9729080, -18.9874325, 27.4399891, -53.7091484, 56.9603424
4: -23.6085434, 35.7925186, -17.0107269, 25.7165833, -49.3251266, 52.8032379

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3496305, upper bound: 46.3973723
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3496844, upper bound: 46.3979147
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -19.7945900, 35.3213921, -13.3021946, 23.8024693, -43.5970535, 48.5638847
1: -22.6274872, 34.8829269, -15.2836475, 23.8005772, -46.4280548, 50.1665726
2: -22.5204334, 34.1403465, -15.1096706, 23.1136723, -45.6341057, 49.2500153
3: -28.4521999, 40.9773903, -19.4732094, 28.0227852, -56.4749832, 60.4505920
4: -25.6193924, 38.6395912, -17.4002285, 26.2846336, -51.9040260, 56.0398178

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020899
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020898
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -12.7971506, 23.0809841, -44.9366531, 51.2741814
1: -24.9799175, 38.2082138, -14.7185745, 23.0831375, -48.0630569, 52.9267883
2: -24.8422871, 37.3502884, -14.5681829, 22.4129333, -47.2552185, 51.9184685
3: -31.4298725, 44.9164658, -18.7758389, 27.1741924, -58.6040649, 63.6922989
4: -28.2795525, 42.3808212, -16.8335400, 25.4575272, -53.7370796, 59.2143555

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3868779, upper bound: 46.4012453
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3868779, upper bound: 46.4012453
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -21.7094116, 38.2469978, -12.2123003, 22.0635204, -43.7729340, 50.4592934
1: -24.8219280, 37.9755707, -14.0449352, 22.0455666, -46.8674927, 52.0205002
2: -24.6786175, 37.1231232, -13.8646946, 21.3987408, -46.0773544, 50.9878159
3: -31.2300510, 44.6412430, -17.9215183, 25.9386024, -57.1686478, 62.5627518
4: -28.0933685, 42.1199799, -15.9714890, 24.3215561, -52.4149246, 58.0914688

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3866803, upper bound: 46.3993625
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3825844, upper bound: 46.3829158
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -12.8695917, 23.1978874, -44.8262749, 50.9881744
1: -24.7276039, 37.7699165, -14.8005753, 23.1950188, -47.9226227, 52.5704880
2: -24.5516720, 36.9445229, -14.6490326, 22.5233936, -47.0750656, 51.5935516
3: -31.0978374, 44.3794708, -18.8776188, 27.3054333, -58.4032555, 63.2570877
4: -27.8886814, 41.8913345, -16.9203701, 25.5841904, -53.4728699, 58.8117065

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3891114, upper bound: 46.4012386
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3891114, upper bound: 46.4012386
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -21.4741402, 37.8610115, -12.2885456, 22.1907845, -43.6649246, 50.1495590
1: -24.5510635, 37.5095062, -14.1313658, 22.1690311, -46.7200851, 51.6408615
2: -24.3705120, 36.6906548, -13.9499454, 21.5208340, -45.8913383, 50.6405983
3: -30.8752117, 44.0720291, -18.0284634, 26.0830059, -56.9582062, 62.1004906
4: -27.6808262, 41.6019783, -16.0642853, 24.4581051, -52.1389236, 57.6662636

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3889137, upper bound: 46.3993558
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848179, upper bound: 46.3829091
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -22.1651039, 38.9499435, -59.0065498, 57.8527412
1: -22.9604015, 35.3986816, -25.3353462, 38.6690903, -61.6294899, 60.7340279
2: -22.8718414, 34.6248589, -25.1847992, 37.8050613, -60.6137276, 59.7522278
3: -28.8905430, 41.6071968, -31.8695679, 45.4547615, -74.3453064, 73.4767609
4: -26.0635490, 39.2244759, -28.6375313, 42.9099388, -68.9734879, 67.8620071

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3528689
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3952419
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -20.0963440, 35.7420425, -21.9534016, 38.6161308, -58.6730385, 57.6192627
1: -22.9604015, 35.3986816, -25.0934258, 38.2545433, -61.2149391, 60.4921074
2: -22.8718414, 34.6248589, -24.9084492, 37.4230347, -60.2303352, 59.4764442
3: -28.8905430, 41.6071968, -31.5502472, 44.9459229, -73.8364639, 73.1574402
4: -26.0635490, 39.2244759, -28.2657051, 42.4474640, -68.5110168, 67.4901810

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3814338
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3952419
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -18.2608814, 32.8123665, -21.8467903, 38.4970627, -56.6598549, 54.5689049
1: -20.8757095, 32.3502274, -24.9734821, 38.1313248, -58.9779968, 57.3237076
2: -20.7539730, 31.6704140, -24.7971897, 37.3032417, -57.9889984, 56.4006157
3: -26.2691593, 37.9729080, -31.4060020, 44.8000374, -71.0691986, 69.3788910
4: -23.6085434, 35.7925186, -28.1573582, 42.2925034, -65.8956757, 63.9288673

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3581148, upper bound: 46.4019957
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -19.7945900, 35.3213921, -22.2130566, 39.0177612, -58.7557297, 57.3793297
1: -22.6274872, 34.8829269, -25.3891716, 38.6557007, -61.2831802, 60.2452812
2: -22.5204334, 34.1403465, -25.1929913, 37.8162651, -60.2589455, 59.2141876
3: -28.4521999, 40.9773903, -31.9208298, 45.4162788, -73.8684692, 72.8982239
4: -25.6193924, 38.6395912, -28.5711136, 42.8956490, -68.4811935, 67.1494446

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806845, upper bound: 46.3872834
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3806845, upper bound: 46.4059533
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -20.0697021, 35.7345085, -57.5132294, 58.5098915
1: -24.9799175, 38.2082138, -22.9424095, 35.3080254, -60.2879410, 61.1506233
2: -24.8422871, 37.3502884, -22.8217659, 34.5563393, -59.3324242, 60.1151886
3: -31.4298725, 44.9164658, -28.8461037, 41.4776268, -72.9075012, 73.7625656
4: -28.2795525, 42.3808212, -25.9452019, 39.1169739, -67.3965302, 68.3260193

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3985079, upper bound: 46.3591697
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4043731, upper bound: 46.3895546
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -20.1475525, 35.8621483, -57.4055748, 58.2408867
1: -24.7276039, 37.7699165, -23.0306740, 35.4290886, -60.1566925, 60.8005905
2: -24.5516720, 36.9445229, -22.9083958, 34.6756935, -59.1699104, 59.8027878
3: -31.0978374, 44.3794708, -28.9547653, 41.6189651, -72.7167969, 73.3342133
4: -27.8886814, 41.8913345, -26.0378189, 39.2533302, -67.1386490, 67.9291534

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3856572, upper bound: 46.3843485
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4029914, upper bound: 46.3895491
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -21.8274021, 38.4213867, -21.8556671, 38.4770317, -60.2673683, 60.2501755
1: -24.9548664, 38.0781250, -24.9799175, 38.2082138, -63.1630745, 63.0580444
2: -24.7688732, 37.2457466, -24.8422871, 37.3502884, -62.0763474, 62.0367699
3: -31.3835411, 44.7412338, -31.4298725, 44.9164658, -76.3000031, 76.1710968
4: -28.1235542, 42.2339706, -28.2795525, 42.3808212, -70.5043716, 70.5135193

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3880580, upper bound: 46.4077136
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4051129, upper bound: 46.4105233
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -21.6283913, 38.1185913, -59.9959908, 60.1396103
1: -25.0402546, 38.1956673, -24.7276039, 37.7699165, -62.8101692, 62.9232712
2: -24.8525200, 37.3617020, -24.5516720, 36.9445229, -61.7611961, 61.8717804
3: -31.4891014, 44.8785400, -31.0978374, 44.3794708, -75.8685608, 75.9763794
4: -28.2129040, 42.3660202, -27.8886814, 41.8913345, -70.1042404, 70.2546997

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4023832, upper bound: 46.3880816
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4049241, upper bound: 46.4076063
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.67 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3418199
IS_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3470623
IS_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3720594, upper bound: 46.3394178
IS_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3818160, upper bound: 46.3470623
IS_A1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3710746
IS_A1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3710746
IS_A1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3739221
IS_A1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3746640
IS_A1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3513339, upper bound: 46.3693564
IS_A1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3746895, upper bound: 46.3926650
IS_A1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
IS_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3956852, upper bound: 46.3989951
IS_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3768760, upper bound: 46.3634878
IS_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3855924, upper bound: 46.3695266
IS_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3741748, upper bound: 46.3899860
IS_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3741748, upper bound: 46.3939982
IS_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3912741, upper bound: 46.3490166
IS_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3917556, upper bound: 46.3491975
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3912741, upper bound: 46.3490166
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3917556, upper bound: 46.3491975
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3973723, upper bound: 46.3496303
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3979147, upper bound: 46.3496843
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3797874
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3769216
IS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4012453, upper bound: 46.3868779
IS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4012453, upper bound: 46.3868779
IS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3993626, upper bound: 46.3866803
IS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3829158, upper bound: 46.3825844
IS_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4012386, upper bound: 46.3891114
IS_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4012386, upper bound: 46.3891114
IS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3993558, upper bound: 46.3889137
IS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3829091, upper bound: 46.3848179
IS_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3490165, upper bound: 46.3912741
IS_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3491975, upper bound: 46.3917556
IS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3490165, upper bound: 46.3912741
IS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3491975, upper bound: 46.3917556
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3496305, upper bound: 46.3973723
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3496844, upper bound: 46.3979147
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020899
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020898
IS_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3868779, upper bound: 46.4012453
IS_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3868779, upper bound: 46.4012453
IS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3866803, upper bound: 46.3993625
IS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3825844, upper bound: 46.3829158
IS_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3891114, upper bound: 46.4012386
IS_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3891114, upper bound: 46.4012386
IS_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3889137, upper bound: 46.3993558
IS_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3848179, upper bound: 46.3829091
IS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3528689
IS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3952419
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3814338
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3952419
IS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
IS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3581148, upper bound: 46.4019957
IS_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3806845, upper bound: 46.3872834
IS_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3806845, upper bound: 46.4059533
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3985079, upper bound: 46.3591697
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4043731, upper bound: 46.3895546
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3856572, upper bound: 46.3843485
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4029914, upper bound: 46.3895491
IS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.3880580, upper bound: 46.4077136
IS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4051129, upper bound: 46.4105233
IS_A2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4023832, upper bound: 46.3880816
IS_A2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 3, lower bound: -46.4049241, upper bound: 46.4076063

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -11.2782097, 20.7617130, -11.2742805, 20.7554340, -32.0336418, 32.0359955
1: -12.9586926, 20.7419739, -12.9542065, 20.7355442, -33.6942368, 33.6961784
2: -12.9154234, 20.1310997, -12.9108934, 20.1249809, -33.0403938, 33.0419846
3: -16.5916309, 24.4350243, -16.5860004, 24.4271832, -41.0188141, 41.0210266
4: -15.0208540, 22.8207054, -15.0160170, 22.8133278, -37.8341789, 37.8367233

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3246968, upper bound: 46.3283489
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3418199
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.8240356, 23.1475544, -11.2742805, 20.7554340, -33.5794678, 34.4218369
1: -14.7506380, 23.2159920, -12.9542065, 20.7355442, -35.4861832, 36.1701965
2: -14.6458588, 22.5191956, -12.9108934, 20.1249809, -34.7708359, 35.4300880
3: -18.8241386, 27.3499374, -16.5860004, 24.4271832, -43.2513199, 43.9359360
4: -16.9858208, 25.6044750, -15.0160170, 22.8133278, -39.7991447, 40.6204910

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3246968, upper bound: 46.3283489
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3418199
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.0850973, 22.0172863, -11.2228603, 20.6781406, -32.7632370, 33.2401390
1: -13.9028893, 21.9789410, -12.8944626, 20.6573372, -34.5602264, 34.8733978
2: -13.7949238, 21.3482666, -12.8536806, 20.0496445, -33.8445663, 34.2019463
3: -17.7513885, 25.8828888, -16.5121193, 24.3345299, -42.0859184, 42.3949966
4: -15.9853020, 24.1958923, -14.9540768, 22.7222385, -38.7075424, 39.1499672

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3572336, upper bound: 46.3217355
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3572336, upper bound: 46.3394178
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -13.2511921, 23.8353138, -11.1965218, 20.6206760, -33.8718681, 35.0318336
1: -15.2209454, 23.7801571, -12.8678284, 20.6072674, -35.8282127, 36.6479874
2: -15.0725260, 23.1033230, -12.8226318, 19.9972801, -35.0698013, 35.9259453
3: -19.3951435, 28.0184917, -16.4775429, 24.2764187, -43.6715622, 44.4960327
4: -17.3597660, 26.2480412, -14.9141397, 22.6683216, -40.0280876, 41.1621819

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3662742, upper bound: 46.3297473
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3662742, upper bound: 46.3470623
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -11.2782097, 20.7617130, -11.2217398, 20.6840553, -31.9622631, 31.9834518
1: -12.9586926, 20.7419739, -12.9060535, 20.5976448, -33.5563354, 33.6480179
2: -12.9154234, 20.1310997, -12.8317413, 20.0037327, -32.9191551, 32.9628410
3: -16.5916309, 24.4350243, -16.4788857, 24.2461224, -40.8377533, 40.9139099
4: -15.0208540, 22.8207054, -14.8719177, 22.6551094, -37.6759644, 37.6926231

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3397771, upper bound: 46.3556583
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3437769, upper bound: 46.3693622
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -12.8240356, 23.1475544, -11.2217398, 20.6840553, -33.5080872, 34.3692932
1: -14.7506380, 23.2159920, -12.9060535, 20.5976448, -35.3482819, 36.1220474
2: -14.6458588, 22.5191956, -12.8317413, 20.0037327, -34.6495895, 35.3509369
3: -18.8241386, 27.3499374, -16.4788857, 24.2461224, -43.0702591, 43.8288231
4: -16.9858208, 25.6044750, -14.8719177, 22.6551094, -39.6409302, 40.4763947

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3397771, upper bound: 46.3703382
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3437769, upper bound: 46.3765857
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -11.2217398, 20.6840553, -11.2217398, 20.6840553, -31.9057961, 31.9057961
1: -12.9060535, 20.5976448, -12.9060535, 20.5976448, -33.5036964, 33.5036926
2: -12.8317413, 20.0037327, -12.8317413, 20.0037327, -32.8354721, 32.8354721
3: -16.4788857, 24.2461224, -16.4788857, 24.2461224, -40.7250061, 40.7250061
4: -14.8719177, 22.6551094, -14.8719177, 22.6551094, -37.5270271, 37.5270271

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3168530, upper bound: 46.3167158
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3744010, upper bound: 46.3739221
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -12.7501745, 22.9591522, -11.2217398, 20.6840553, -33.4342232, 34.1808891
1: -14.6577673, 22.9542313, -12.9060535, 20.5976448, -35.2554054, 35.8602791
2: -14.5072327, 22.2842579, -12.8317413, 20.0037327, -34.5109634, 35.1159973
3: -18.6941471, 27.0294781, -16.4788857, 24.2461224, -42.9402695, 43.5083618
4: -16.7502651, 25.3299522, -14.8719177, 22.6551094, -39.4053726, 40.2018700

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3168530, upper bound: 46.3167158
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3744010, upper bound: 46.3739221
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -11.1817913, 20.3261948, -12.3597021, 22.4504948, -33.6322861, 32.6858940
1: -12.8648539, 20.3061886, -14.2198849, 22.5062103, -35.3710594, 34.5260620
2: -12.7136984, 19.6851120, -14.1387548, 21.8302193, -34.5439148, 33.8238640
3: -16.4651089, 23.8802719, -18.1687641, 26.5144997, -42.9796066, 42.0490303
4: -14.6671801, 22.3373642, -16.4323521, 24.8003368, -39.4675179, 38.7697144

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3513339, upper bound: 46.3693564
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3513339, upper bound: 46.3693564
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.8290615, 22.8110409, -12.4238186, 22.5566578, -35.3857193, 35.2348595
1: -14.7397728, 22.8367844, -14.2916946, 22.6156921, -37.3554649, 37.1284790
2: -14.5288477, 22.1392288, -14.2101288, 21.9348812, -36.4637260, 36.3493538
3: -18.8284607, 26.8803425, -18.2638226, 26.6450825, -45.4735413, 45.1441650
4: -16.6528397, 25.2215214, -16.5185108, 24.9253922, -41.5782318, 41.7400208

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3584379, upper bound: 46.3775152
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3700420, upper bound: 46.3889299
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -13.0116901, 23.4255905, -12.8240356, 23.1475544, -36.1592445, 36.2496185
1: -14.9617081, 23.4858780, -14.7506380, 23.2159920, -38.1777000, 38.2365150
2: -14.8465652, 22.7847881, -14.6458588, 22.5191956, -37.3657608, 37.4306374
3: -19.0826035, 27.6671886, -18.8241386, 27.3499374, -46.4325371, 46.4913254
4: -17.1964169, 25.9140949, -16.9858208, 25.6044750, -42.8008919, 42.8999176

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3937666, upper bound: 46.3976518
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.9515820, 23.2615986, -12.8240356, 23.1475544, -36.0991364, 36.0856285
1: -14.8831511, 23.2486610, -14.7506380, 23.2159920, -38.0991440, 37.9992981
2: -14.7253942, 22.5756149, -14.6458588, 22.5191956, -37.2445869, 37.2214699
3: -18.9707508, 27.3725243, -18.8241386, 27.3499374, -46.3206863, 46.1966629
4: -16.9788628, 25.6684685, -16.9858208, 25.6044750, -42.5833359, 42.6542854

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3915564, upper bound: 46.3972800
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -11.9866343, 21.9139957, -11.2899084, 20.6029358, -32.5895691, 33.2039032
1: -13.7936935, 21.8731632, -12.9966555, 20.5821228, -34.3758163, 34.8698158
2: -13.6943674, 21.2441349, -12.8449049, 19.9628601, -33.6572227, 34.0890388
3: -17.6215439, 25.7534008, -16.6375847, 24.2210846, -41.8426285, 42.3909836
4: -15.8902750, 24.0618420, -14.8703861, 22.6440258, -38.5343018, 38.9322281

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3730442, upper bound: 46.3534047
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3730442, upper bound: 46.3634878
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1439342, 23.7158432, -11.2773838, 20.5554295, -33.6993637, 34.9932213
1: -15.1030178, 23.6605225, -12.9802341, 20.5422974, -35.6453171, 36.6407547
2: -14.9629860, 22.9867630, -12.8241434, 19.9199162, -34.8828964, 35.8109055
3: -19.2528419, 27.8704033, -16.6162796, 24.1743908, -43.4272308, 44.4866791
4: -17.2540112, 26.0971050, -14.8387356, 22.6015568, -39.8555641, 40.9358406

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3808560, upper bound: 46.3596250
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3808560, upper bound: 46.3695266
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -11.8803215, 21.5007057, -12.6439228, 22.7910843, -34.6714020, 34.1446304
1: -13.6646070, 21.4798603, -14.5368881, 22.7865276, -36.4511223, 36.0167465
2: -13.4940720, 20.8352280, -14.3889275, 22.1210289, -35.6151009, 35.2241554
3: -17.4712029, 25.2716560, -18.5426521, 26.8304729, -44.3016739, 43.8143082
4: -15.5687332, 23.6612968, -16.6194782, 25.1390991, -40.7078323, 40.2807732

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3473292, upper bound: 46.3681124
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3899860
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1922131, 23.6291485, -12.6439228, 22.7910843, -35.9832993, 36.2730713
1: -15.1584854, 23.6275005, -14.5368881, 22.7865276, -37.9450111, 38.1643906
2: -14.9872761, 22.9454288, -14.3889275, 22.1210289, -37.1083069, 37.3343582
3: -19.3160419, 27.8168468, -18.5426521, 26.8304729, -46.1465149, 46.3594971
4: -17.2645493, 26.0874233, -16.6194782, 25.1390991, -42.4036446, 42.7069016

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3499674, upper bound: 46.3935858
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3937684
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -12.6048870, 22.8473625, -20.0963440, 35.7420425, -48.3469276, 42.9437065
1: -14.5055141, 22.9045258, -22.9604015, 35.3986816, -49.9041939, 45.8649292
2: -14.4079285, 22.2194366, -22.8718414, 34.6248589, -49.0327873, 45.0912743
3: -18.5224590, 26.9783039, -28.8905430, 41.6071968, -60.1296539, 55.8688431
4: -16.7317600, 25.2493343, -26.0635490, 39.2244759, -55.9562302, 51.3128815

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3855367, upper bound: 46.3429997
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966801, upper bound: 46.3506670
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.1694489, 22.1034031, -19.9430447, 35.5074005, -47.6768494, 42.0464439
1: -14.0022125, 22.1406517, -22.7897396, 35.1582222, -49.1604309, 44.9303894
2: -13.8697376, 21.4670792, -22.7016373, 34.3897972, -48.2595253, 44.1687050
3: -17.8813591, 26.0725422, -28.6767311, 41.3223534, -59.2037048, 54.7492599
4: -16.1070061, 24.4025421, -25.8687057, 38.9550056, -55.0620117, 50.2712479

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3865528, upper bound: 46.3429471
time: 1.08 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3969988, upper bound: 46.3511662
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.6297007, 22.8327045, -20.0963440, 35.7420425, -48.3717422, 42.9290466
1: -14.5259962, 22.8189259, -22.9604015, 35.3986816, -49.9246788, 45.7793236
2: -14.3886719, 22.1563683, -22.8718414, 34.6248589, -49.0135307, 45.0282097
3: -18.5333290, 26.8647728, -28.8905430, 41.6071968, -60.1405220, 55.7553177
4: -16.6362762, 25.1681004, -26.0635490, 39.2244759, -55.8607483, 51.2316513

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3778395, upper bound: 46.3414123
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3884612, upper bound: 46.3490166
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.0750809, 21.8547401, -19.9430447, 35.5074005, -47.5592308, 41.7977829
1: -13.8874903, 21.8282700, -22.7897396, 35.1582222, -49.0457115, 44.6180077
2: -13.7144489, 21.1867065, -22.7016373, 34.3897972, -48.1042404, 43.8883400
3: -17.7196064, 25.6817436, -28.6767311, 41.3223534, -59.0419617, 54.3584709
4: -15.8067064, 24.0807076, -25.8687057, 38.9550056, -54.7617111, 49.9494133

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3782512, upper bound: 46.3415439
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3888079, upper bound: 46.3491975
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 7.22 seconds
IS_A1_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3246968, upper bound: 46.3283489
IS_A1_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3418199
IS_A1_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3246968, upper bound: 46.3283489
IS_A1_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3418199
IS_A1_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3572336, upper bound: 46.3217355
IS_A1_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3572336, upper bound: 46.3394178
IS_A1_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3662742, upper bound: 46.3297473
IS_A1_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3662742, upper bound: 46.3470623
IS_A1_B1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3397771, upper bound: 46.3556583
IS_A1_B1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3437769, upper bound: 46.3693622
IS_A1_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3397771, upper bound: 46.3703382
IS_A1_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3437769, upper bound: 46.3765857
IS_A1_B1_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3168530, upper bound: 46.3167158
IS_A1_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3744010, upper bound: 46.3739221
IS_A1_B1_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3168530, upper bound: 46.3167158
IS_A1_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3744010, upper bound: 46.3739221
IS_A1_B1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3513339, upper bound: 46.3693564
IS_A1_B1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3513339, upper bound: 46.3693564
IS_A1_B1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3584379, upper bound: 46.3775152
IS_A1_B1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3700420, upper bound: 46.3889299
IS_A1_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3937666, upper bound: 46.3976518
IS_A1_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
IS_A1_B1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3915564, upper bound: 46.3972800
IS_A1_B1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3920027, upper bound: 46.3974541
IS_A1_B1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3730442, upper bound: 46.3534047
IS_A1_B1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3730442, upper bound: 46.3634878
IS_A1_B1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3808560, upper bound: 46.3596250
IS_A1_B1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3808560, upper bound: 46.3695266
IS_A1_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3473292, upper bound: 46.3681124
IS_A1_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3899860
IS_A1_B1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3499674, upper bound: 46.3935858
IS_A1_B1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3708136, upper bound: 46.3937684
IS_A1_B2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3855367, upper bound: 46.3429997
IS_A1_B2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3966801, upper bound: 46.3506670
IS_A1_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3865528, upper bound: 46.3429471
IS_A1_B2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3969988, upper bound: 46.3511662
IS_A1_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3778395, upper bound: 46.3414123
IS_A1_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3884612, upper bound: 46.3490166
IS_A1_B2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3782512, upper bound: 46.3415439
IS_A1_B2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3888079, upper bound: 46.3491975
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3973723, upper bound: 46.3496303
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3979147, upper bound: 46.3496843
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3797874
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3934067, upper bound: 46.3769216
IS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4012453, upper bound: 46.3868779
IS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4012453, upper bound: 46.3868779
IS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3993626, upper bound: 46.3866803
IS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3829158, upper bound: 46.3825844
IS_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4012386, upper bound: 46.3891114
IS_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4012386, upper bound: 46.3891114
IS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3993558, upper bound: 46.3889137
IS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3829091, upper bound: 46.3848179
IS_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3490165, upper bound: 46.3912741
IS_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3491975, upper bound: 46.3917556
IS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3490165, upper bound: 46.3912741
IS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3491975, upper bound: 46.3917556
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3496305, upper bound: 46.3973723
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3496844, upper bound: 46.3979147
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020899
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3797874, upper bound: 46.4020898
IS_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3868779, upper bound: 46.4012453
IS_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3868779, upper bound: 46.4012453
IS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3866803, upper bound: 46.3993625
IS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3825844, upper bound: 46.3829158
IS_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3891114, upper bound: 46.4012386
IS_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3891114, upper bound: 46.4012386
IS_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3889137, upper bound: 46.3993558
IS_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3848179, upper bound: 46.3829091
IS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3528689
IS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3952419
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3814338
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3952419
IS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3581148, upper bound: 46.3846116
IS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3581148, upper bound: 46.4019957
IS_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3806845, upper bound: 46.3872834
IS_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3806845, upper bound: 46.4059533
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3985079, upper bound: 46.3591697
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4043731, upper bound: 46.3895546
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3856572, upper bound: 46.3843485
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4029914, upper bound: 46.3895491
IS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.3880580, upper bound: 46.4077136
IS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4051129, upper bound: 46.4105233
IS_A2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4023832, upper bound: 46.3880816
IS_A2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 3, lower bound: -46.4049241, upper bound: 46.4076063
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257
time: 0.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -46.4091342, upper bound: 46.4024257

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -15.5593853, 27.1077271, -40.4099159, 39.3618546
1: -15.2836475, 23.8005772, -17.8337097, 27.1816788, -42.4653244, 41.6342773
2: -15.1096706, 23.1136723, -17.5771866, 26.3973064, -41.5069771, 40.6908569
3: -19.4732094, 28.0227852, -22.6461716, 31.9880333, -51.4612236, 50.6689568
4: -17.4002285, 26.2846336, -20.0349903, 30.2143135, -47.6145363, 46.3196259

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

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
time: 0.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -15.6051130, 27.1780758, -49.3911247, 54.6228714
1: -25.3891716, 38.6557007, -17.8864689, 27.2607346, -52.6499062, 56.5421677
2: -25.1929913, 37.8162651, -17.6271324, 26.4738178, -51.6667976, 55.4433975
3: -31.9208298, 45.4162788, -22.7172165, 32.0807762, -64.0016022, 68.1334839
4: -28.5711136, 42.8956490, -20.0892334, 30.3068199, -58.8779335, 62.9848824

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

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
- Time for IS candidates: 4.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 3, lower bound: -46.4007247, upper bound: 46.4007247
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 3, lower bound: -46.4007247, upper bound: 46.4024257
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 3, lower bound: -46.4024257, upper bound: 46.4091342
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 3, lower bound: -46.4024257, upper bound: 46.4091342

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -13.3021946, 23.8024693, -37.1046600, 37.1046600
1: -15.2836475, 23.8005772, -15.2836475, 23.8005772, -39.0842171, 39.0842209
2: -15.1096706, 23.1136723, -15.1096706, 23.1136723, -38.2233429, 38.2233429
3: -19.4732094, 28.0227852, -19.4732094, 28.0227852, -47.4959869, 47.4959869
4: -17.4002285, 26.2846336, -17.4002285, 26.2846336, -43.6848602, 43.6848602

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3771602, upper bound: 46.3920130
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999558, upper bound: 46.3999560
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.3021946, 23.8024693, -22.2130566, 39.0177612, -52.3199539, 46.0155220
1: -15.2836475, 23.8005772, -25.3891716, 38.6557007, -53.9393463, 49.1897392
2: -15.1096706, 23.1136723, -25.1929913, 37.8162651, -52.9259338, 48.3066559
3: -19.4732094, 28.0227852, -31.9208298, 45.4162788, -64.8894882, 59.9436111
4: -17.4002285, 26.2846336, -28.5711136, 42.8956490, -60.2958755, 54.8557472

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920130, upper bound: 46.3801372
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999560, upper bound: 46.4017220
time: 0.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -13.3021946, 23.8024693, -46.0155220, 52.3199539
1: -25.3891716, 38.6557007, -15.2836475, 23.8005772, -49.1897392, 53.9393463
2: -25.1929913, 37.8162651, -15.1096706, 23.1136723, -48.3066559, 52.9259338
3: -31.9208298, 45.4162788, -19.4732094, 28.0227852, -59.9436111, 64.8894806
4: -28.5711136, 42.8956490, -17.4002285, 26.2846336, -54.8557472, 60.2958755

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3771602, upper bound: 46.4032098
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3771602, upper bound: 46.4083758
time: 0.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.2130566, 39.0177612, -22.2130566, 39.0177612, -61.2015877, 61.2015839
1: -25.3891716, 38.6557007, -25.3891716, 38.6557007, -64.0448761, 64.0448761
2: -25.1929913, 37.8162651, -25.1929913, 37.8162651, -62.9735107, 62.9735184
3: -31.9208298, 45.4162788, -31.9208298, 45.4162788, -77.3370972, 77.3370972
4: -28.5711136, 42.8956490, -28.5711136, 42.8956490, -71.4667664, 71.4667664

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920130, upper bound: 46.3771602
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3920130, upper bound: 46.3859390
time: 0.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.14 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3771602, upper bound: 46.3920130
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3999558, upper bound: 46.3999560
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3920130, upper bound: 46.3801372
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3999560, upper bound: 46.4017220
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3771602, upper bound: 46.4032098
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3771602, upper bound: 46.4083758
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3920130, upper bound: 46.3771602
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -46.3920130, upper bound: 46.3859390

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.3869448, 20.9255638, -13.1098967, 23.4882927, -34.8752365, 34.0354614
1: -13.0953932, 20.8397293, -15.0614872, 23.4775543, -36.5729446, 35.9012146
2: -13.0110359, 20.2390823, -14.8939400, 22.8004951, -35.8115196, 35.1330185
3: -16.7173176, 24.5306568, -19.1864395, 27.6395149, -44.3568344, 43.7170944
4: -15.0642700, 22.9335842, -17.1458912, 25.9256058, -40.9898720, 40.0794754

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3771602, upper bound: 46.3920130
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3760730, upper bound: 46.3900698
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.0015049, 23.3439465, -13.2792902, 23.7677631, -36.7692680, 36.6232376
1: -14.9456024, 23.3515148, -15.2579060, 23.7663574, -38.7119598, 38.6094170
2: -14.7805185, 22.6720467, -15.0845919, 23.0800953, -37.8606071, 37.7566376
3: -19.0547180, 27.4934063, -19.4412346, 27.9824276, -47.0371475, 46.9346390
4: -17.0490189, 25.7683182, -17.3733730, 26.2452545, -43.2942696, 43.1416817

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947109, upper bound: 46.3983184
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.1098967, 23.4882927, -20.1475525, 35.8621483, -48.9720459, 43.6358452
1: -15.0614872, 23.4775543, -23.0306740, 35.4290886, -50.4905777, 46.5082283
2: -14.8939400, 22.8004951, -22.9083958, 34.6756935, -49.5696335, 45.7088852
3: -19.1864395, 27.6395149, -28.9547653, 41.6189651, -60.8054008, 56.5942802
4: -17.1458912, 25.9256058, -26.0378189, 39.2533302, -56.3992233, 51.9634209

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3765187
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3765187
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.2792902, 23.7677631, -21.9027481, 38.5440521, -51.8233414, 45.6705093
1: -15.2579060, 23.7663574, -25.0402546, 38.1956673, -53.4535751, 48.8066101
2: -15.0845919, 23.0800953, -24.8525200, 37.3617020, -52.4462929, 47.9326096
3: -19.4412346, 27.9824276, -31.4891014, 44.8785400, -64.3197784, 59.4715271
4: -17.3733730, 26.2452545, -28.2129040, 42.3660202, -59.7393913, 54.4581566

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033613, upper bound: 46.4002133
time: 0.89 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033369, upper bound: 46.3970053
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -13.1098967, 23.4882927, -43.6358452, 48.9720459
1: -23.0306740, 35.4290886, -15.0614872, 23.4775543, -46.5082283, 50.4905777
2: -22.9083958, 34.6756935, -14.8939400, 22.8004951, -45.7088814, 49.5696297
3: -28.9547653, 41.6189651, -19.1864395, 27.6395149, -56.5942802, 60.8054047
4: -26.0378189, 39.2533302, -17.1458912, 25.9256058, -51.9634209, 56.3992233

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3843905
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3920130
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.9027481, 38.5440521, -13.2792902, 23.7677631, -45.6705093, 51.8233414
1: -25.0402546, 38.1956673, -15.2579060, 23.7663574, -48.8066101, 53.4535751
2: -24.8525200, 37.3617020, -15.0845919, 23.0800953, -47.9326096, 52.4462929
3: -31.4891014, 44.8785400, -19.4412346, 27.9824276, -59.4715271, 64.3197784
4: -28.2129040, 42.3660202, -17.3733730, 26.2452545, -54.4581566, 59.7393913

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033613
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3970053, upper bound: 46.4033367
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -21.9798317, 38.6418304, -20.1475525, 35.8621483, -57.7569847, 58.7463646
1: -25.1221142, 38.2726822, -23.0306740, 35.4290886, -60.5511971, 61.3033485
2: -24.9317474, 37.4446640, -22.9083958, 34.6756935, -59.5530205, 60.2864113
3: -31.5767937, 44.9638023, -28.9547653, 41.6189651, -73.1957474, 73.9185486
4: -28.2709656, 42.4668617, -26.0378189, 39.2533302, -67.5113144, 68.4994125

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3796969, upper bound: 46.3568122
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006051, upper bound: 46.3893424
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -22.1897621, 38.9822578, -21.9027481, 38.5440521, -60.7032928, 60.8565292
1: -25.3629589, 38.6212311, -25.0402546, 38.1956673, -63.5586243, 63.6614838
2: -25.1673756, 37.7821846, -24.8525200, 37.3617020, -62.4918404, 62.5978699
3: -31.8884945, 45.3759727, -31.4891014, 44.8785400, -76.7670212, 76.8650742
4: -28.5440598, 42.8559570, -28.2129040, 42.3660202, -70.9100647, 71.0688629

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4048811, upper bound: 46.4104510
time: 3.40 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074626
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.89 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3771602, upper bound: 46.3920130
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3760730, upper bound: 46.3900698
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3947109, upper bound: 46.3983184
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3765187
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3765187
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.4033613, upper bound: 46.4002133
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.4033369, upper bound: 46.3970053
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3843905
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3735154, upper bound: 46.3920130
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033613
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3970053, upper bound: 46.4033367
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.3796969, upper bound: 46.3568122
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.4006051, upper bound: 46.3893424
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.4048811, upper bound: 46.4104510
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.89
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074626

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.1909981, 20.6098518, -12.9533634, 23.3185501, -34.5095482, 33.5632172
1: -12.8724174, 20.5394917, -14.8926630, 23.3711338, -36.2435493, 35.4321480
2: -12.7918577, 19.9426193, -14.7795248, 22.6741333, -35.4659920, 34.7221451
3: -16.4411068, 24.1776810, -18.9874725, 27.5325241, -43.9736252, 43.1651535
4: -14.8273668, 22.5903797, -17.1069393, 25.7922440, -40.6196098, 39.6973190

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3731644, upper bound: 46.3902552
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3762386
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3900698
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.3817081, 20.9176979, -12.8699093, 23.1210785, -34.5027847, 33.7876053
1: -13.0893526, 20.8313885, -14.7869377, 23.0989075, -36.1882591, 35.6183243
2: -13.0053835, 20.2312870, -14.6331415, 22.4306183, -35.4359970, 34.8644218
3: -16.7097454, 24.5212402, -18.8418694, 27.1988163, -43.9085617, 43.3631058
4: -15.0581388, 22.9244347, -16.8629532, 25.5070934, -40.5652275, 39.7873878

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3762386
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3900698
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7877016, 23.0021000, -13.0966072, 23.5605526, -36.3482513, 36.0987091
1: -14.7037067, 23.0233536, -15.0585518, 23.6209259, -38.3246307, 38.0818977
2: -14.5424566, 22.3489017, -14.9414301, 22.9162788, -37.4587326, 37.2903290
3: -18.7542725, 27.1080437, -19.2039433, 27.8281307, -46.5824051, 46.3119812
4: -16.7932034, 25.3961372, -17.3020134, 26.0678196, -42.8610229, 42.6981468

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947109, upper bound: 46.3983162
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3940143, upper bound: 46.3967917
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.9949265, 23.3338432, -13.0362062, 23.3963776, -36.3913040, 36.3700485
1: -14.9380779, 23.3411293, -14.9796209, 23.3831902, -38.3212662, 38.3207436
2: -14.7733440, 22.6619034, -14.8200912, 22.7062454, -37.4795876, 37.4819908
3: -19.0453014, 27.4811993, -19.0921459, 27.5329933, -46.5782928, 46.5733452
4: -17.0411625, 25.7568493, -17.0843048, 25.8217888, -42.8629532, 42.8411560

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -11.3869448, 20.9255638, -20.1475525, 35.8621483, -47.2490921, 41.0731163
1: -13.0953932, 20.8397293, -23.0306740, 35.4290886, -48.5244789, 43.8704033
2: -13.0110359, 20.2390823, -22.9083958, 34.6756935, -47.6867256, 43.1474762
3: -16.7173176, 24.5306568, -28.9547653, 41.6189651, -58.3362808, 53.4854202
4: -15.0642700, 22.9335842, -26.0378189, 39.2533302, -54.3176003, 48.9714012

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3841794, upper bound: 46.3763162
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3682838, upper bound: 46.3749404
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.0015049, 23.3439465, -20.1475525, 35.8621483, -48.8636551, 43.4915009
1: -14.9456024, 23.3515148, -23.0306740, 35.4290886, -50.3746910, 46.3821793
2: -14.7805185, 22.6720467, -22.9083958, 34.6756935, -49.4562073, 45.5804443
3: -19.0547180, 27.4934063, -28.9547653, 41.6189651, -60.6736794, 56.4481735
4: -17.0490189, 25.7683182, -26.0378189, 39.2533302, -56.3023491, 51.8061295

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3717118, upper bound: 46.3763162
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3682838, upper bound: 46.3749404
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -13.0551729, 23.4074459, -21.8556671, 38.4770317, -51.5322037, 45.2631149
1: -15.0043058, 23.4208851, -24.9799175, 38.2082138, -53.2125130, 48.4008026
2: -14.8342581, 22.7397537, -24.8422871, 37.3502884, -52.1845398, 47.5820351
3: -19.1265697, 27.5770206, -31.4298725, 44.9164658, -64.0430222, 59.0068893
4: -17.1042614, 25.8537102, -28.2795525, 42.3808212, -59.4850769, 54.1332626

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033235, upper bound: 46.4002133
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033235, upper bound: 46.4002133
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.2725401, 23.7575016, -21.6283913, 38.1185913, -51.3911324, 45.3858948
1: -15.2501831, 23.7556896, -24.7276039, 37.7699165, -53.0200882, 48.4832916
2: -15.0772486, 23.0697441, -24.5516720, 36.9445229, -52.0217552, 47.6214142
3: -19.4314747, 27.9698677, -31.0978374, 44.3794708, -63.8109436, 59.0677032
4: -17.3652325, 26.2334499, -27.8886814, 41.8913345, -59.2565689, 54.1221313

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033191, upper bound: 46.3970053
time: 1.06 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033189, upper bound: 46.3970053
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -11.3869448, 20.9255638, -41.0731163, 47.2490921
1: -23.0306740, 35.4290886, -13.0953932, 20.8397293, -43.8704033, 48.5244789
2: -22.9083958, 34.6756935, -13.0110359, 20.2390823, -43.1474762, 47.6867256
3: -28.9547653, 41.6189651, -16.7173176, 24.5306568, -53.4854202, 58.3362808
4: -26.0378189, 39.2533302, -15.0642700, 22.9335842, -48.9714012, 54.3176003

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3692960, upper bound: 46.3717118
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3682838, upper bound: 46.3682838
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -20.1475525, 35.8621483, -13.0015049, 23.3439465, -43.4915009, 48.8636551
1: -23.0306740, 35.4290886, -14.9456024, 23.3515148, -46.3821831, 50.3746910
2: -22.9083958, 34.6756935, -14.7805185, 22.6720467, -45.5804405, 49.4562073
3: -28.9547653, 41.6189651, -19.0547180, 27.4934063, -56.4481735, 60.6736832
4: -26.0378189, 39.2533302, -17.0490189, 25.7683182, -51.8061295, 56.3023491

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3692960, upper bound: 46.3717118
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3749404, upper bound: 46.3808135
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -21.8556671, 38.4770317, -13.0551729, 23.4074459, -45.2631149, 51.5322037
1: -24.9799175, 38.2082138, -15.0043058, 23.4208851, -48.4008026, 53.2125168
2: -24.8422871, 37.3502884, -14.8342581, 22.7397537, -47.5820389, 52.1845398
3: -31.4298725, 44.9164658, -19.1265697, 27.5770206, -59.0068893, 64.0430222
4: -28.2795525, 42.3808212, -17.1042614, 25.8537102, -54.1332626, 59.4850769

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033235
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033235
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -21.6283913, 38.1185913, -13.2725401, 23.7575016, -45.3858948, 51.3911324
1: -24.7276039, 37.7699165, -15.2501831, 23.7556896, -48.4832916, 53.0200882
2: -24.5516720, 36.9445229, -15.0772486, 23.0697441, -47.6214104, 52.0217590
3: -31.0978374, 44.3794708, -19.4314747, 27.9698677, -59.0676994, 63.8109360
4: -27.8886814, 41.8913345, -17.3652325, 26.2334499, -54.1221313, 59.2565689

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3970053, upper bound: 46.4033189
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3724384, upper bound: 46.3932603
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -21.7378025, 38.2471924, -20.0963440, 35.7420425, -57.4066315, 58.2783241
1: -24.8479710, 37.8947487, -22.9604015, 35.3986816, -60.2466507, 60.8551483
2: -24.6624508, 37.0718155, -22.8718414, 34.6248589, -59.2343102, 59.8557625
3: -31.2380257, 44.5223885, -28.8905430, 41.6071968, -72.8451996, 73.4129257
4: -27.9827633, 42.0416832, -26.0635490, 39.2244759, -67.2050781, 68.1052246

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3796969, upper bound: 46.3568122
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3796969, upper bound: 46.3568122
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -21.9729290, 38.6309319, -19.9100819, 35.4986420, -57.3822632, 58.4944725
1: -25.1142445, 38.2617836, -22.7595940, 35.0610962, -60.1753349, 61.0213776
2: -24.9241791, 37.4339828, -22.6484280, 34.3139229, -59.1848984, 60.0121422
3: -31.5669518, 44.9510345, -28.6167450, 41.1884537, -72.7554016, 73.5677567
4: -28.2627716, 42.4547768, -25.7594795, 38.8430481, -67.0898209, 68.2088089

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006051, upper bound: 46.3893370
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3900698, upper bound: 46.3760730
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -21.9496269, 38.5909882, -21.8556671, 38.4770317, -60.3903542, 60.4120750
1: -25.0910416, 38.2467651, -24.9799175, 38.2082138, -63.2992554, 63.2266769
2: -24.9004288, 37.4126434, -24.8422871, 37.3502884, -62.2102966, 62.1979980
3: -31.5526028, 44.9385643, -31.4298725, 44.9164658, -76.4690704, 76.3684387
4: -28.2590637, 42.4347687, -28.2795525, 42.3808212, -70.6398773, 70.7143250

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074625
time: 0.91 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074626
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -22.1827736, 38.9712448, -21.6283913, 38.1185913, -60.2751465, 60.5682182
1: -25.3549919, 38.6102066, -24.7276039, 37.7699165, -63.1249084, 63.3378029
2: -25.1597042, 37.7713776, -24.5516720, 36.9445229, -62.0692291, 62.2828827
3: -31.8785210, 45.3630638, -31.0978374, 44.3794708, -76.2579803, 76.4608994
4: -28.5357494, 42.8437347, -27.8886814, 41.8913345, -70.4270782, 70.7324142

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932603, upper bound: 46.3932601
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074626
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.26 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3762386
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3900698
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3762386
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3900698
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3947109, upper bound: 46.3983162
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3940143, upper bound: 46.3967917
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3841794, upper bound: 46.3763162
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3682838, upper bound: 46.3749404
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3717118, upper bound: 46.3763162
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3682838, upper bound: 46.3749404
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4033235, upper bound: 46.4002133
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4033235, upper bound: 46.4002133
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4033191, upper bound: 46.3970053
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4033189, upper bound: 46.3970053
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3692960, upper bound: 46.3717118
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3682838, upper bound: 46.3682838
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3692960, upper bound: 46.3717118
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3749404, upper bound: 46.3808135
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033235
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033235
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3970053, upper bound: 46.4033189
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3724384, upper bound: 46.3932603
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3796969, upper bound: 46.3568122
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3796969, upper bound: 46.3568122
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4006051, upper bound: 46.3893370
IS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3900698, upper bound: 46.3760730
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074625
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074626
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.3932603, upper bound: 46.3932601
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -46.4046585, upper bound: 46.4074626

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.2742805, 20.7554340, -12.9533634, 23.3185501, -34.5928307, 33.7087975
1: -12.9542065, 20.7355442, -14.8926630, 23.3711338, -36.3253403, 35.6282082
2: -12.9108934, 20.1249809, -14.7795248, 22.6741333, -35.5850258, 34.9045067
3: -16.5860004, 24.4271832, -18.9874725, 27.5325241, -44.1185226, 43.4146576
4: -15.0160170, 22.8133278, -17.1069393, 25.7922440, -40.8082619, 39.9202652

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3376083, upper bound: 46.3745977
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3481961, upper bound: 46.3853819
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.2217398, 20.6840553, -12.9533634, 23.3185501, -34.5402908, 33.6374207
1: -12.9060535, 20.5976448, -14.8926630, 23.3711338, -36.2771797, 35.4903030
2: -12.8317413, 20.0037327, -14.7795248, 22.6741333, -35.5058746, 34.7832565
3: -16.4788857, 24.2461224, -18.9874725, 27.5325241, -44.0114098, 43.2335968
4: -14.8719177, 22.6551094, -17.1069393, 25.7922440, -40.6641617, 39.7620468

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3376083, upper bound: 46.3791106
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3481961, upper bound: 46.3902006
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.2742805, 20.7554340, -12.8699093, 23.1210785, -34.3953590, 33.6253433
1: -12.9542065, 20.7355442, -14.7869377, 23.0989075, -36.0531158, 35.5224800
2: -12.9108934, 20.1249809, -14.6331415, 22.4306183, -35.3415031, 34.7581139
3: -16.5860004, 24.4271832, -18.8418694, 27.1988163, -43.7848167, 43.2690506
4: -15.0160170, 22.8133278, -16.8629532, 25.5070934, -40.5231056, 39.6762810

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3645429
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3418199, upper bound: 46.3762386
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.2217398, 20.6840553, -12.8699093, 23.1210785, -34.3428192, 33.5539627
1: -12.9060535, 20.5976448, -14.7869377, 23.0989075, -36.0049553, 35.3845787
2: -12.8317413, 20.0037327, -14.6331415, 22.4306183, -35.2623558, 34.6368675
3: -16.4788857, 24.2461224, -18.8418694, 27.1988163, -43.6777039, 43.0879898
4: -14.8719177, 22.6551094, -16.8629532, 25.5070934, -40.3790131, 39.5180626

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3356996, upper bound: 46.3765433
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3462688, upper bound: 46.3865032
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.4736938, 22.4562511, -12.4751320, 22.4894314, -34.9631195, 34.9313812
1: -14.3429098, 22.4691925, -14.3467598, 22.5339794, -36.8768806, 36.8159523
2: -14.1811562, 21.8150806, -14.2290535, 21.8666496, -36.0477982, 36.0441360
3: -18.3006439, 26.4489021, -18.3069019, 26.5353069, -44.8359451, 44.7558060
4: -16.3639717, 24.7857037, -16.4649315, 24.8614140, -41.2253876, 41.2506332

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947109, upper bound: 46.3983162
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3947109, upper bound: 46.3983162
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.6461029, 22.7607784, -14.0390511, 24.8432350, -37.4893379, 36.7998276
1: -14.5428743, 22.7810440, -16.1272488, 24.9246178, -39.4674911, 38.9082947
2: -14.3817158, 22.1122856, -15.9351902, 24.1863823, -38.5681000, 38.0474663
3: -18.5525036, 26.8204288, -20.5496864, 29.3725204, -47.9250259, 47.3701134
4: -16.6097736, 25.1318302, -18.3426533, 27.5866890, -44.1964607, 43.4744835

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3940143, upper bound: 46.3967917
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3940143, upper bound: 46.3967917
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.8240356, 23.1475544, -13.0362062, 23.3963776, -36.2204094, 36.1837616
1: -14.7506380, 23.2159920, -14.9796209, 23.3831902, -38.1338272, 38.1956139
2: -14.6458588, 22.5191956, -14.8200912, 22.7062454, -37.3520966, 37.3392830
3: -18.8241386, 27.3499374, -19.0921459, 27.5329933, -46.3571320, 46.4420815
4: -16.9858208, 25.6044750, -17.0843048, 25.8217888, -42.8076096, 42.6887817

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.7501745, 22.9591522, -13.0362062, 23.3963776, -36.1465492, 35.9953575
1: -14.6577673, 22.9542313, -14.9796209, 23.3831902, -38.0409584, 37.9338455
2: -14.5072327, 22.2842579, -14.8200912, 22.7062454, -37.2134705, 37.1043472
3: -18.6941471, 27.0294781, -19.0921459, 27.5329933, -46.2271385, 46.1216240
4: -16.7502651, 25.3299522, -17.0843048, 25.8217888, -42.5720520, 42.4142570

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3932601, upper bound: 46.3932603
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -10.8425064, 19.9800720, -19.8125381, 35.3111153, -46.1453400, 39.7926025
1: -12.4706945, 19.8719921, -22.6489105, 34.8635635, -47.3342590, 42.5209045
2: -12.3833370, 19.3048630, -22.5274582, 34.1273193, -46.5106544, 41.8323212
3: -15.9254370, 23.3743553, -28.4752464, 40.9493561, -56.8747940, 51.8496017
4: -14.3148079, 21.8611050, -25.6033058, 38.6253357, -52.9401398, 47.4644089

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3567448, upper bound: 46.3627164
time: 1.29 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814182, upper bound: 46.3736902
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.3501654, 22.2573509, -19.9707394, 35.5671883, -47.9173546, 42.2280884
1: -14.1879454, 22.1911659, -22.8298264, 35.1312408, -49.3191872, 45.0209923
2: -14.0256996, 21.5479412, -22.7093067, 34.3867035, -48.4123993, 44.2572479
3: -18.0946980, 26.1247597, -28.7032127, 41.2667389, -59.3614311, 54.8279724
4: -16.1306133, 24.5090122, -25.8116989, 38.9226379, -55.0532532, 50.3207092

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3470163, upper bound: 46.3637179
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3764974, upper bound: 46.3719393
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.3775806, 22.2650738, -19.8125381, 35.3111153, -47.6677399, 42.0775986
1: -14.2303123, 22.2562962, -22.6489105, 34.8635635, -49.0938759, 44.9052048
2: -14.0636492, 21.6173172, -22.5274582, 34.1273193, -48.1909676, 44.1447754
3: -18.1563988, 26.1904507, -28.4752464, 40.9493561, -59.1057549, 54.6656952
4: -16.1992435, 24.5620365, -25.6033058, 38.6253357, -54.8245697, 50.1653442

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3767858, upper bound: 46.3673177
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4004477, upper bound: 46.3775002
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -13.9856091, 24.7094440, -19.9707394, 35.5671883, -49.5527954, 44.6801758
1: -16.0607777, 24.7400284, -22.8298264, 35.1312408, -51.1920166, 47.5698509
2: -15.8340168, 24.0266819, -22.7093067, 34.3867035, -50.2207146, 46.7359848
3: -20.4685383, 29.1352978, -28.7032127, 41.2667389, -61.7352753, 57.8385086
4: -18.1372814, 27.3947010, -25.8116989, 38.9226379, -57.0599213, 53.2063980

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4015839, upper bound: 46.3791065
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014654, upper bound: 46.3791221
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -12.4351540, 22.3340969, -21.5065517, 37.9056358, -50.3407860, 43.8406487
1: -14.2941236, 22.3326664, -24.5879478, 37.6286697, -51.9227905, 46.9206161
2: -14.1221647, 21.6910973, -24.4510365, 36.7883568, -50.9105186, 46.1421318
3: -18.2337360, 26.2827511, -30.9373150, 44.2294922, -62.4632263, 57.2200661
4: -16.2616959, 24.6549187, -27.8327637, 41.7356834, -57.9973793, 52.4876823

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3842858, upper bound: 46.3911791
time: 0.93 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989109, upper bound: 46.3986026
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -14.0362053, 24.7682915, -21.6861420, 38.1958504, -52.2320557, 46.4544334
1: -16.1168499, 24.8075848, -24.7889977, 37.9239388, -54.0407829, 49.5965805
2: -15.8883762, 24.0915508, -24.6523209, 37.0753365, -52.9636993, 48.7438698
3: -20.5363083, 29.2166004, -31.1899452, 44.5798798, -65.1161880, 60.4065475
4: -18.1917725, 27.4770508, -28.0632496, 42.0656700, -60.2574348, 55.5402946

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033235, upper bound: 46.4002133
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4033235, upper bound: 46.4002133
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -12.6449184, 22.6745720, -21.2791595, 37.5355911, -50.1805077, 43.9537315
1: -14.5315514, 22.6570263, -24.3282642, 37.1787148, -51.7102661, 46.9852905
2: -14.3571558, 22.0116081, -24.1542358, 36.3712959, -50.7284508, 46.1658401
3: -18.5283871, 26.6634045, -30.5971146, 43.6796188, -62.2080078, 57.2605209
4: -16.5143967, 25.0230141, -27.4341984, 41.2363167, -57.7507133, 52.4572029

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3835108, upper bound: 46.3864002
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989109, upper bound: 46.3943352
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -14.2464495, 25.1070747, -21.4421673, 37.8096542, -52.0561028, 46.5492401
1: -16.3548431, 25.1304131, -24.5155525, 37.4579239, -53.8127632, 49.6459656
2: -16.1223507, 24.4106655, -24.3418350, 36.6423149, -52.7646599, 48.7524872
3: -20.8318596, 29.5952930, -30.8320084, 44.0105362, -64.8423843, 60.4272995
4: -18.4435825, 27.8438931, -27.6504250, 41.5449066, -59.9884872, 55.4943123

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4027720, upper bound: 46.3952857
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4027720, upper bound: 46.3970053
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -19.8125381, 35.3111153, -10.8425064, 19.9800720, -39.7926064, 46.1453400
1: -22.6489105, 34.8635635, -12.4706945, 19.8719921, -42.5209045, 47.3342590
2: -22.5274582, 34.1273193, -12.3833370, 19.3048630, -41.8323174, 46.5106583
3: -28.4752464, 40.9493561, -15.9254370, 23.3743553, -51.8495979, 56.8747940
4: -25.6033058, 38.6253357, -14.3148079, 21.8611050, -47.4644089, 52.9401360

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3571248, upper bound: 46.3567456
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3736902, upper bound: 46.3814181
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -19.9707394, 35.5671883, -12.3501654, 22.2573509, -42.2280884, 47.9173546
1: -22.8298264, 35.1312408, -14.1879454, 22.1911659, -45.0209923, 49.3191872
2: -22.7093067, 34.3867035, -14.0256996, 21.5479412, -44.2572479, 48.4123993
3: -28.7032127, 41.2667389, -18.0946980, 26.1247597, -54.8279724, 59.3614273
4: -25.8116989, 38.9226379, -16.1306133, 24.5090122, -50.3207092, 55.0532532

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3572039, upper bound: 46.3512808
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3719393, upper bound: 46.3764974
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -19.8125381, 35.3111153, -12.3775806, 22.2650738, -42.0775986, 47.6677322
1: -22.6489105, 34.8635635, -14.2303123, 22.2562962, -44.9052048, 49.0938759
2: -22.5274582, 34.1273193, -14.0636492, 21.6173172, -44.1447754, 48.1909676
3: -28.4752464, 40.9493561, -18.1563988, 26.1904507, -54.6656952, 59.1057549
4: -25.6033058, 38.6253357, -16.1992435, 24.5620365, -50.1653442, 54.8245659

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3673177, upper bound: 46.3767858
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3775002, upper bound: 46.4004477
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -19.9707394, 35.5671883, -13.9856091, 24.7094440, -44.6801758, 49.5527954
1: -22.8298264, 35.1312408, -16.0607777, 24.7400284, -47.5698509, 51.1920128
2: -22.7093067, 34.3867035, -15.8340168, 24.0266819, -46.7359848, 50.2207184
3: -28.7032127, 41.2667389, -20.4685383, 29.1352978, -57.8385086, 61.7352753
4: -25.8116989, 38.9226379, -18.1372814, 27.3947010, -53.2063980, 57.0599213

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3791065, upper bound: 46.4015842
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3791232, upper bound: 46.4014654
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5065517, 37.9056358, -12.4351540, 22.3340969, -43.8406487, 50.3407860
1: -24.5879478, 37.6286697, -14.2941236, 22.3326664, -46.9206161, 51.9227905
2: -24.4510365, 36.7883568, -14.1221647, 21.6910973, -46.1421318, 50.9105186
3: -30.9373150, 44.2294922, -18.2337360, 26.2827511, -57.2200661, 62.4632263
4: -27.8327637, 41.7356834, -16.2616959, 24.6549187, -52.4876823, 57.9973793

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3911791, upper bound: 46.3842858
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986026, upper bound: 46.3989109
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -21.6861420, 38.1958504, -14.0362053, 24.7682915, -46.4544334, 52.2320557
1: -24.7889977, 37.9239388, -16.1168499, 24.8075848, -49.5965805, 54.0407867
2: -24.6523209, 37.0753365, -15.8883762, 24.0915508, -48.7438660, 52.9636993
3: -31.1899452, 44.5798798, -20.5363083, 29.2166004, -60.4065475, 65.1161880
4: -28.0632496, 42.0656700, -18.1917725, 27.4770508, -55.5402985, 60.2574348

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033235
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4002133, upper bound: 46.4033235
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -21.2791595, 37.5355911, -12.6449184, 22.6745720, -43.9537315, 50.1805077
1: -24.3282642, 37.1787148, -14.5315514, 22.6570263, -46.9852905, 51.7102661
2: -24.1542358, 36.3712959, -14.3571558, 22.0116081, -46.1658401, 50.7284508
3: -30.5971146, 43.6796188, -18.5283871, 26.6634045, -57.2605209, 62.2080078
4: -27.4341984, 41.2363167, -16.5143967, 25.0230141, -52.4572029, 57.7507133

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3864002, upper bound: 46.3835108
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3943352, upper bound: 46.3989109
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -21.4421673, 37.8096542, -14.2464495, 25.1070747, -46.5492401, 52.0561028
1: -24.5155525, 37.4579239, -16.3548431, 25.1304131, -49.6459656, 53.8127632
2: -24.3418350, 36.6423149, -16.1223507, 24.4106655, -48.7524872, 52.7646637
3: -30.8320084, 44.0105362, -20.8318596, 29.5952930, -60.4272995, 64.8423843
4: -27.6504250, 41.5449066, -18.4435825, 27.8438931, -55.4943085, 59.9884872

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3952857, upper bound: 46.4027720
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3952857, upper bound: 46.4027720
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -21.9579124, 38.6124382, -20.0963440, 35.7420425, -57.6455040, 58.6559601
1: -25.0994778, 38.3252029, -22.9604015, 35.3986816, -60.4981613, 61.2855988
2: -24.9531803, 37.4717293, -22.8718414, 34.6248589, -59.5203362, 60.2664871
3: -31.5651398, 45.0490456, -28.8905430, 41.6071968, -73.1723328, 73.9395828
4: -28.3680420, 42.5264702, -26.0635490, 39.2244759, -67.5925140, 68.5900192

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3528689
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3568122
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -21.7201691, 38.2391167, -20.0963440, 35.7420425, -57.3838005, 58.2686195
1: -24.8262463, 37.8703918, -22.9604015, 35.3986816, -60.2249298, 60.8307915
2: -24.6472282, 37.0504074, -22.8718414, 34.6248589, -59.2138557, 59.8328972
3: -31.2057858, 44.4922829, -28.8905430, 41.6071968, -72.8129807, 73.3828278
4: -27.9651337, 42.0178757, -26.0635490, 39.2244759, -67.1863937, 68.0776520

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3562873
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3517948, upper bound: 46.3568122
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -21.9579124, 38.6124382, -19.9100819, 35.4986420, -57.3665543, 58.4692078
1: -25.0994778, 38.3252029, -22.7595940, 35.0610962, -60.1605721, 61.0847969
2: -24.9531803, 37.4717293, -22.6484280, 34.3139229, -59.1966782, 60.0461006
3: -31.5651398, 45.0490456, -28.6167450, 41.1884537, -72.7535858, 73.6657486
4: -28.3680420, 42.5264702, -25.7594795, 38.8430481, -67.2044220, 68.2859497

Time for backsubstitution: 2.55 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=55.00176239013672
rel_dist={3: [-46.41123857032521, 46.41123857032903]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1117.84 seconds
