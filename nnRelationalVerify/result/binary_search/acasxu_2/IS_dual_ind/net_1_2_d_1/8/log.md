## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 43.3827531155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307)
1: (-8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275)
2: (-9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216)
3: (-14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580)
4: (-14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412)

## BASE Result
execution time: IAR + LP analysis = 2.02 + 1.49 = 3.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569


# Binary Search by BASE starts (time budget: 1196.49 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=46.890357971191406
rel_dist={3: [-43.60068373509955, 43.60068373509955]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=46.890357971191406
rel_dist={3: [-43.6005733842848, 43.6005733842848]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=46.890357971191406
rel_dist={3: [-43.60049262479107, 43.60049262479107]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=46.890357971191406
rel_dist={3: [-43.600442096761945, 43.60044209676194]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=46.890357971191406
rel_dist={3: [-43.600415482278315, 43.60041548227832]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=46.890357971191406
rel_dist={3: [-43.60040197566741, 43.60040197566741]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=46.890357971191406
rel_dist={3: [-43.600395222369386, 43.600395222369386]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=46.890357971191406
rel_dist={3: [-43.600391845735146, 43.600391845735146]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=46.890357971191406
rel_dist={3: [-43.60039015744708, 43.60039015744708]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=46.890357971191406
rel_dist={3: [-43.60038931335932, 43.60038931335933]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=46.890357971191406
rel_dist={3: [-43.60038889142116, 43.60038889142115]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=46.890357971191406
rel_dist={3: [-43.60038868063948, 43.60038868930823]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=46.890357971191406
rel_dist={3: [-43.60038857723265, 43.60038857813507]}

## Binary Search Result
Binary search time: 63.24 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1133.26 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5965478, upper bound: 43.5783800
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 3, lower bound: -43.5965478, upper bound: 43.5783800
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.6606002, 28.3044319, -35.7576599, 35.2311859
1: -8.7061062, 31.1410255, -8.9601765, 31.9744511, -40.6805573, 40.1012039
2: -9.2308626, 31.3365688, -9.4893942, 32.1866379, -41.4174995, 40.8259621
3: -13.6493845, 31.9803791, -14.0368652, 32.8534927, -46.5028763, 46.0172424
4: -14.1980610, 31.2951145, -14.5853920, 32.1437492, -46.3418121, 45.8805046

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.45 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.6606002, 28.3044319, -35.9425278, 35.9483070
1: -8.9388247, 31.9677963, -8.9601765, 31.9744511, -40.9132729, 40.9279709
2: -9.4615984, 32.1662331, -9.4893942, 32.1866379, -41.6482315, 41.6556206
3: -14.0235004, 32.8585358, -14.0368652, 32.8534927, -46.8769913, 46.8954010
4: -14.5844440, 32.0941963, -14.5853920, 32.1437492, -46.7281952, 46.6795807

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.4532280, 27.5705853, -35.0238037, 35.0238037
1: -8.7061062, 31.1410255, -8.7061062, 31.1410255, -39.8471298, 39.8471298
2: -9.2308626, 31.3365688, -9.2308626, 31.3365688, -40.5674324, 40.5674324
3: -13.6493845, 31.9803791, -13.6493845, 31.9803791, -45.6297646, 45.6297646
4: -14.1980610, 31.2951145, -14.1980610, 31.2951145, -45.4931755, 45.4931755

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5899906, upper bound: 43.5321191
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.6380959, 28.2877064, -35.7409325, 35.2086716
1: -8.7061062, 31.1410255, -8.9388247, 31.9677963, -40.6739006, 40.0798416
2: -9.2308626, 31.3365688, -9.4615984, 32.1662331, -41.3970947, 40.7981682
3: -13.6493845, 31.9803791, -14.0235004, 32.8585358, -46.5079193, 46.0038795
4: -14.1980610, 31.2951145, -14.5844440, 32.0941963, -46.2922554, 45.8795586

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5899906, upper bound: 43.5321191
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.51 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.4532280, 27.5705853, -35.2086716, 35.7409363
1: -8.9388247, 31.9677963, -8.7061062, 31.1410255, -40.0798416, 40.6739044
2: -9.4615984, 32.1662331, -9.2308626, 31.3365688, -40.7981682, 41.3970947
3: -14.0235004, 32.8585358, -13.6493845, 31.9803791, -46.0038795, 46.5079193
4: -14.5844440, 32.0941963, -14.1980610, 31.2951145, -45.8795586, 46.2922554

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.53 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.6380959, 28.2877064, -35.9258003, 35.9258003
1: -8.9388247, 31.9677963, -8.9388247, 31.9677963, -40.9066124, 40.9066124
2: -9.4615984, 32.1662331, -9.4615984, 32.1662331, -41.6278305, 41.6278305
3: -14.0235004, 32.8585358, -14.0235004, 32.8585358, -46.8820343, 46.8820343
4: -14.5844440, 32.0941963, -14.5844440, 32.0941963, -46.6786385, 46.6786385

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.05 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5899906, upper bound: 43.5321191
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5899906, upper bound: 43.5321191
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.4532280, 27.5705853, -34.7577667, 34.0631332
1: -8.3826809, 30.0424767, -8.7061062, 31.1410255, -39.5237045, 38.7485809
2: -8.9014492, 30.2400932, -9.2308626, 31.3365688, -40.2380180, 39.4709549
3: -13.1529007, 30.8414726, -13.6493845, 31.9803791, -45.1332779, 44.4908562
4: -13.6967449, 30.2116623, -14.1980610, 31.2951145, -44.9918594, 44.4097214

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.4532280, 27.5705853, -35.3001633, 35.8521881
1: -9.0717134, 32.0508652, -8.7061062, 31.1410255, -40.2127304, 40.7569733
2: -9.5743036, 32.3409538, -9.2308626, 31.3365688, -40.9108734, 41.5718155
3: -14.1834040, 32.9562950, -13.6493845, 31.9803791, -46.1637726, 46.6056786
4: -14.6434193, 32.4114075, -14.1980610, 31.2951145, -45.9385338, 46.6094666

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.6380959, 28.2877064, -35.4748917, 34.2480011
1: -8.3826809, 30.0424767, -8.9388247, 31.9677963, -40.3504791, 38.9812927
2: -8.9014492, 30.2400932, -9.4615984, 32.1662331, -41.0676804, 39.7016907
3: -13.1529007, 30.8414726, -14.0235004, 32.8585358, -46.0114365, 44.8649750
4: -13.6967449, 30.2116623, -14.5844440, 32.0941963, -45.7909393, 44.7961044

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.6380959, 28.2877064, -36.0172882, 36.0370560
1: -9.0717134, 32.0508652, -8.9388247, 31.9677963, -41.0395050, 40.9896851
2: -9.5743036, 32.3409538, -9.4615984, 32.1662331, -41.7405319, 41.8025513
3: -14.1834040, 32.9562950, -14.0235004, 32.8585358, -47.0419350, 46.9797974
4: -14.6434193, 32.4114075, -14.5844440, 32.0941963, -46.7376099, 46.9958496

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.4532280, 27.5705853, -34.9407043, 34.7802124
1: -8.6132603, 30.8662567, -8.7061062, 31.1410255, -39.7542839, 39.5723572
2: -9.1305141, 31.0644073, -9.2308626, 31.3365688, -40.4670830, 40.2952690
3: -13.5239582, 31.7180500, -13.6493845, 31.9803791, -45.5043373, 45.3674355
4: -14.0812511, 31.0096245, -14.1980610, 31.2951145, -45.3763618, 45.2076836

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.4532280, 27.5705853, -35.4747047, 36.5295334
1: -9.2882051, 32.8371925, -8.7061062, 31.1410255, -40.4292297, 41.5432930
2: -9.7911968, 33.1104507, -9.2308626, 31.3365688, -41.1277657, 42.3413124
3: -14.5323257, 33.7886276, -13.6493845, 31.9803791, -46.5126991, 47.4380112
4: -15.0085173, 33.1586647, -14.1980610, 31.2951145, -46.3036270, 47.3567276

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.6380959, 28.2877064, -35.6578293, 34.9650764
1: -8.6132603, 30.8662567, -8.9388247, 31.9677963, -40.5810547, 39.8050690
2: -9.1305141, 31.0644073, -9.4615984, 32.1662331, -41.2967453, 40.5260048
3: -13.5239582, 31.7180500, -14.0235004, 32.8585358, -46.3824921, 45.7415504
4: -14.0812511, 31.0096245, -14.5844440, 32.0941963, -46.1754379, 45.5940704

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.6380959, 28.2877064, -36.1918259, 36.7144012
1: -9.2882051, 32.8371925, -8.9388247, 31.9677963, -41.2560005, 41.7760048
2: -9.7911968, 33.1104507, -9.4615984, 32.1662331, -41.9574280, 42.5720482
3: -14.5323257, 33.7886276, -14.0235004, 32.8585358, -47.3908577, 47.8121262
4: -15.0085173, 33.1586647, -14.5844440, 32.0941963, -47.1027031, 47.7431107

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.10 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.1871862, 26.6099091, -33.7970886, 33.7970886
1: -8.3826809, 30.0424767, -8.3826809, 30.0424767, -38.4251556, 38.4251556
2: -8.9014492, 30.2400932, -8.9014492, 30.2400932, -39.1415405, 39.1415405
3: -13.1529007, 30.8414726, -13.1529007, 30.8414726, -43.9943733, 43.9943733
4: -13.6967449, 30.2116623, -13.6967449, 30.2116623, -43.9084091, 43.9084091

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.7295799, 28.3989639, -35.5861473, 34.3394890
1: -8.3826809, 30.0424767, -9.0717134, 32.0508652, -40.4335480, 39.1141815
2: -8.9014492, 30.2400932, -9.5743036, 32.3409538, -41.2424011, 39.8143959
3: -13.1529007, 30.8414726, -14.1834040, 32.9562950, -46.1091957, 45.0248756
4: -13.6967449, 30.2116623, -14.6434193, 32.4114075, -46.1081543, 44.8550797

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.1871862, 26.6099091, -34.3394890, 35.5861473
1: -9.0717134, 32.0508652, -8.3826809, 30.0424767, -39.1141815, 40.4335480
2: -9.5743036, 32.3409538, -8.9014492, 30.2400932, -39.8143959, 41.2424011
3: -14.1834040, 32.9562950, -13.1529007, 30.8414726, -45.0248756, 46.1091957
4: -14.6434193, 32.4114075, -13.6967449, 30.2116623, -44.8550797, 46.1081543

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.7295799, 28.3989639, -36.1285439, 36.1285439
1: -9.0717134, 32.0508652, -9.0717134, 32.0508652, -41.1225739, 41.1225739
2: -9.5743036, 32.3409538, -9.5743036, 32.3409538, -41.9152565, 41.9152565
3: -14.1834040, 32.9562950, -14.1834040, 32.9562950, -47.1396904, 47.1396904
4: -14.6434193, 32.4114075, -14.6434193, 32.4114075, -47.0548248, 47.0548248

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.3701210, 27.3269844, -34.5141716, 33.9800301
1: -8.3826809, 30.0424767, -8.6132603, 30.8662567, -39.2489357, 38.6557350
2: -8.9014492, 30.2400932, -9.1305141, 31.0644073, -39.9658546, 39.3706055
3: -13.1529007, 30.8414726, -13.5239582, 31.7180500, -44.8709488, 44.3654327
4: -13.6967449, 30.2116623, -14.0812511, 31.0096245, -44.7063675, 44.2929077

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5876652, upper bound: 43.5290275
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.9041195, 29.0763073, -36.2634926, 34.5140305
1: -8.3826809, 30.0424767, -9.2882051, 32.8371925, -41.2198677, 39.3306808
2: -8.9014492, 30.2400932, -9.7911968, 33.1104507, -42.0119019, 40.0312881
3: -13.1529007, 30.8414726, -14.5323257, 33.7886276, -46.9415283, 45.3737984
4: -13.6967449, 30.2116623, -15.0085173, 33.1586647, -46.8554077, 45.2201729

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5876652, upper bound: 43.5290275
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.3701210, 27.3269844, -35.0565643, 35.7690849
1: -9.0717134, 32.0508652, -8.6132603, 30.8662567, -39.9379578, 40.6641235
2: -9.5743036, 32.3409538, -9.1305141, 31.0644073, -40.6387062, 41.4714661
3: -14.1834040, 32.9562950, -13.5239582, 31.7180500, -45.9014473, 46.4802551
4: -14.6434193, 32.4114075, -14.0812511, 31.0096245, -45.6530418, 46.4926529

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.9041195, 29.0763073, -36.8058853, 36.3030815
1: -9.0717134, 32.0508652, -9.2882051, 32.8371925, -41.9088936, 41.3390694
2: -9.5743036, 32.3409538, -9.7911968, 33.1104507, -42.6847534, 42.1321487
3: -14.1834040, 32.9562950, -14.5323257, 33.7886276, -47.9720230, 47.4886169
4: -14.6434193, 32.4114075, -15.0085173, 33.1586647, -47.8020859, 47.4199181

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.1871862, 26.6099091, -33.9800262, 34.5141678
1: -8.6132603, 30.8662567, -8.3826809, 30.0424767, -38.6557350, 39.2489357
2: -9.1305141, 31.0644073, -8.9014492, 30.2400932, -39.3706055, 39.9658546
3: -13.5239582, 31.7180500, -13.1529007, 30.8414726, -44.3654327, 44.8709488
4: -14.0812511, 31.0096245, -13.6967449, 30.2116623, -44.2929077, 44.7063675

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.7295799, 28.3989639, -35.7690849, 35.0565643
1: -8.6132603, 30.8662567, -9.0717134, 32.0508652, -40.6641235, 39.9379578
2: -9.1305141, 31.0644073, -9.5743036, 32.3409538, -41.4714661, 40.6387062
3: -13.5239582, 31.7180500, -14.1834040, 32.9562950, -46.4802551, 45.9014473
4: -14.0812511, 31.0096245, -14.6434193, 32.4114075, -46.4926529, 45.6530418

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.1871862, 26.6099091, -34.5140305, 36.2634926
1: -9.2882051, 32.8371925, -8.3826809, 30.0424767, -39.3306808, 41.2198677
2: -9.7911968, 33.1104507, -8.9014492, 30.2400932, -40.0312881, 42.0119019
3: -14.5323257, 33.7886276, -13.1529007, 30.8414726, -45.3737984, 46.9415283
4: -15.0085173, 33.1586647, -13.6967449, 30.2116623, -45.2201729, 46.8554077

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.7295799, 28.3989639, -36.3030853, 36.8058853
1: -9.2882051, 32.8371925, -9.0717134, 32.0508652, -41.3390694, 41.9088936
2: -9.7911968, 33.1104507, -9.5743036, 32.3409538, -42.1321487, 42.6847534
3: -14.5323257, 33.7886276, -14.1834040, 32.9562950, -47.4886169, 47.9720230
4: -15.0085173, 33.1586647, -14.6434193, 32.4114075, -47.4199181, 47.8020859

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.3701210, 27.3269844, -34.6971054, 34.6971054
1: -8.6132603, 30.8662567, -8.6132603, 30.8662567, -39.4795113, 39.4795113
2: -9.1305141, 31.0644073, -9.1305141, 31.0644073, -40.1949158, 40.1949196
3: -13.5239582, 31.7180500, -13.5239582, 31.7180500, -45.2420082, 45.2420082
4: -14.0812511, 31.0096245, -14.0812511, 31.0096245, -45.0908699, 45.0908699

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.9041195, 29.0763073, -36.4464264, 35.2311020
1: -8.6132603, 30.8662567, -9.2882051, 32.8371925, -41.4504471, 40.1544571
2: -9.1305141, 31.0644073, -9.7911968, 33.1104507, -42.2409668, 40.8556061
3: -13.5239582, 31.7180500, -14.5323257, 33.7886276, -47.3125839, 46.2503738
4: -14.0812511, 31.0096245, -15.0085173, 33.1586647, -47.2399139, 46.0181351

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.3701210, 27.3269844, -35.2311020, 36.4464264
1: -9.2882051, 32.8371925, -8.6132603, 30.8662567, -40.1544571, 41.4504471
2: -9.7911968, 33.1104507, -9.1305141, 31.0644073, -40.8556061, 42.2409668
3: -14.5323257, 33.7886276, -13.5239582, 31.7180500, -46.2503738, 47.3125839
4: -15.0085173, 33.1586647, -14.0812511, 31.0096245, -46.0181351, 47.2399139

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.9041195, 29.0763073, -36.9804268, 36.9804268
1: -9.2882051, 32.8371925, -9.2882051, 32.8371925, -42.1253929, 42.1253929
2: -9.7911968, 33.1104507, -9.7911968, 33.1104507, -42.9016495, 42.9016495
3: -14.5323257, 33.7886276, -14.5323257, 33.7886276, -48.3209496, 48.3209496
4: -15.0085173, 33.1586647, -15.0085173, 33.1586647, -48.1671791, 48.1671791

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.14 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5876652, upper bound: 43.5290275
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5876652, upper bound: 43.5290275
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.1871862, 26.6099091, -33.3433685, 32.2624207
1: -7.8301010, 28.3211842, -8.3826809, 30.0424767, -37.8725777, 36.7038651
2: -8.3398962, 28.4661274, -8.9014492, 30.2400932, -38.5799904, 37.3675766
3: -12.3204012, 29.0548534, -13.1529007, 30.8414726, -43.1618729, 42.2077560
4: -12.9191322, 28.3510303, -13.6967449, 30.2116623, -43.1307907, 42.0477753

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.1871862, 26.6099091, -33.9370155, 34.4277458
1: -8.5321484, 30.8015041, -8.3826809, 30.0424767, -38.5746231, 39.1841850
2: -9.0758047, 30.9408817, -8.9014492, 30.2400932, -39.3158951, 39.8423309
3: -13.4046860, 31.6406326, -13.1529007, 30.8414726, -44.2461586, 44.7935295
4: -14.0712585, 30.7746468, -13.6967449, 30.2116623, -44.2829056, 44.4713898

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.7295799, 28.3989639, -35.1324234, 32.8048172
1: -7.8301010, 28.3211842, -9.0717134, 32.0508652, -39.8809662, 37.3928909
2: -8.3398962, 28.4661274, -9.5743036, 32.3409538, -40.6808510, 38.0404320
3: -12.3204012, 29.0548534, -14.1834040, 32.9562950, -45.2766953, 43.2382507
4: -12.9191322, 28.3510303, -14.6434193, 32.4114075, -45.3305359, 42.9944496

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.7295799, 28.3989639, -35.7260742, 34.9701462
1: -8.5321484, 30.8015041, -9.0717134, 32.0508652, -40.5830154, 39.8732147
2: -9.0758047, 30.9408817, -9.5743036, 32.3409538, -41.4167557, 40.5151863
3: -13.4046860, 31.6406326, -14.1834040, 32.9562950, -46.3609810, 45.8240280
4: -14.0712585, 30.7746468, -14.6434193, 32.4114075, -46.4826508, 45.4180641

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5691259, upper bound: 43.5588638
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.1871862, 26.6099091, -33.1162109, 31.4548264
1: -7.5842962, 27.4015598, -8.3826809, 30.0424767, -37.6267662, 35.7842407
2: -8.0552320, 27.5818462, -8.9014492, 30.2400932, -38.2953262, 36.4832954
3: -11.9514847, 28.1189060, -13.1529007, 30.8414726, -42.7929573, 41.2718048
4: -12.5255804, 27.4433079, -13.6967449, 30.2116623, -42.7372398, 41.1400528

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.1840873, 26.5985279, -35.5233459, 40.3694344
1: -10.5382071, 37.6687279, -8.3789291, 30.0294685, -40.5676613, 46.0476570
2: -11.0602732, 37.7767830, -8.8976927, 30.2270870, -41.2873611, 46.6744728
3: -16.4940701, 38.9501419, -13.1470404, 30.8277702, -47.3218384, 52.0971832
4: -17.3545589, 37.3720055, -13.6906538, 30.1988678, -47.5534286, 51.0626602

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.7295799, 28.3989639, -34.9052658, 31.9972191
1: -7.5842962, 27.4015598, -9.0717134, 32.0508652, -39.6351585, 36.4732628
2: -8.0552320, 27.5818462, -9.5743036, 32.3409538, -40.3961868, 37.1561470
3: -11.9514847, 28.1189060, -14.1834040, 32.9562950, -44.9077797, 42.3023033
4: -12.5255804, 27.4433079, -14.6434193, 32.4114075, -44.9369850, 42.0867271

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.7270870, 28.3896866, -37.3145065, 40.9124336
1: -10.5382071, 37.6687279, -9.0686817, 32.0401726, -42.5783730, 46.7374115
2: -11.0602732, 37.7767830, -9.5712700, 32.3304024, -43.3906746, 47.3480530
3: -16.4940701, 38.9501419, -14.1786652, 32.9450378, -49.4391098, 53.1288071
4: -17.3545589, 37.3720055, -14.6384068, 32.4011078, -49.7556686, 52.0104141

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.3701210, 27.3269844, -34.0604477, 32.4453583
1: -7.8301010, 28.3211842, -8.6132603, 30.8662567, -38.6963577, 36.9344444
2: -8.3398962, 28.4661274, -9.1305141, 31.0644073, -39.4043045, 37.5966415
3: -12.3204012, 29.0548534, -13.5239582, 31.7180500, -44.0384521, 42.5788116
4: -12.9191322, 28.3510303, -14.0812511, 31.0096245, -43.9287529, 42.4322815

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.3701210, 27.3269844, -34.6540985, 34.6106834
1: -8.5321484, 30.8015041, -8.6132603, 30.8662567, -39.3984032, 39.4147644
2: -9.0758047, 30.9408817, -9.1305141, 31.0644073, -40.1402054, 40.0713959
3: -13.4046860, 31.6406326, -13.5239582, 31.7180500, -45.1227341, 45.1645889
4: -14.0712585, 30.7746468, -14.0812511, 31.0096245, -45.0808716, 44.8558922

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5672675, upper bound: 43.5546301
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.9041195, 29.0763073, -35.8097687, 32.9793549
1: -7.8301010, 28.3211842, -9.2882051, 32.8371925, -40.6672935, 37.6093903
2: -8.3398962, 28.4661274, -9.7911968, 33.1104507, -41.4503479, 38.2573242
3: -12.3204012, 29.0548534, -14.5323257, 33.7886276, -46.1090279, 43.5871773
4: -12.9191322, 28.3510303, -15.0085173, 33.1586647, -46.0777969, 43.3595467

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.9041195, 29.0763073, -36.4034271, 35.1446838
1: -8.5321484, 30.8015041, -9.2882051, 32.8371925, -41.3693390, 40.0897102
2: -9.0758047, 30.9408817, -9.7911968, 33.1104507, -42.1862564, 40.7320786
3: -13.4046860, 31.6406326, -14.5323257, 33.7886276, -47.1933136, 46.1729546
4: -14.0712585, 30.7746468, -15.0085173, 33.1586647, -47.2299118, 45.7831573

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5653596, upper bound: 43.5237682
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5623864, upper bound: 43.5188589
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.3701210, 27.3269844, -33.8332863, 31.6377602
1: -7.5842962, 27.4015598, -8.6132603, 30.8662567, -38.4505424, 36.0148125
2: -8.0552320, 27.5818462, -9.1305141, 31.0644073, -39.1196404, 36.7123566
3: -11.9514847, 28.1189060, -13.5239582, 31.7180500, -43.6695328, 41.6428642
4: -12.5255804, 27.4433079, -14.0812511, 31.0096245, -43.5352020, 41.5245514

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.3670039, 27.3154736, -36.2402954, 40.5523491
1: -10.5382071, 37.6687279, -8.6094589, 30.8530769, -41.3912849, 46.2781868
2: -11.0602732, 37.7767830, -9.1267338, 31.0512180, -42.1114922, 46.9035187
3: -16.4940701, 38.9501419, -13.5180254, 31.7041950, -48.1982651, 52.4681664
4: -17.3545589, 37.3720055, -14.0751152, 30.9966831, -48.3512421, 51.4471169

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.9041195, 29.0763073, -35.5826111, 32.1717606
1: -7.5842962, 27.4015598, -9.2882051, 32.8371925, -40.4214783, 36.6897659
2: -8.0552320, 27.5818462, -9.7911968, 33.1104507, -41.1656837, 37.3730431
3: -11.9514847, 28.1189060, -14.5323257, 33.7886276, -45.7401123, 42.6512299
4: -12.5255804, 27.4433079, -15.0085173, 33.1586647, -45.6842461, 42.4518204

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.9010825, 29.0650730, -37.9898949, 41.0864334
1: -10.5382071, 37.6687279, -9.2845058, 32.8242683, -43.3624763, 46.9532318
2: -11.0602732, 37.7767830, -9.7874994, 33.0975723, -44.1578445, 47.5642815
3: -16.4940701, 38.9501419, -14.5265522, 33.7750397, -50.2691116, 53.4766922
4: -17.3545589, 37.3720055, -15.0024967, 33.1459808, -50.5005417, 52.3745041

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.1871862, 26.6099091, -32.7415123, 30.3283634
1: -7.1131840, 26.1596413, -8.3826809, 30.0424767, -37.1556625, 34.5423203
2: -7.5950131, 26.2545033, -8.9014492, 30.2400932, -37.8351059, 35.1559525
3: -11.2687273, 26.8291836, -13.1529007, 30.8414726, -42.1101990, 39.9820786
4: -11.9468098, 25.9775772, -13.6967449, 30.2116623, -42.1584702, 39.6743240

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.1840873, 26.5985279, -35.1313400, 39.2040710
1: -10.0470028, 36.3811836, -8.3789291, 30.0294685, -40.0764656, 44.7601128
2: -10.5810204, 36.3961143, -8.8976927, 30.2270870, -40.8081055, 45.2937965
3: -15.7773991, 37.6067963, -13.1470404, 30.8277702, -46.6051712, 50.7538376
4: -16.7466125, 35.8476906, -13.6906538, 30.1988678, -46.9454803, 49.5383453

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.7295799, 28.3989639, -34.5305672, 30.8707561
1: -7.1131840, 26.1596413, -9.0717134, 32.0508652, -39.1640472, 35.2313538
2: -7.5950131, 26.2545033, -9.5743036, 32.3409538, -39.9359665, 35.8288040
3: -11.2687273, 26.8291836, -14.1834040, 32.9562950, -44.2250214, 41.0125771
4: -11.9468098, 25.9775772, -14.6434193, 32.4114075, -44.3582153, 40.6209946

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.7270870, 28.3896866, -36.9225044, 39.7470703
1: -10.0470028, 36.3811836, -9.0686817, 32.0401726, -42.0871735, 45.4498672
2: -10.5810204, 36.3961143, -9.5712700, 32.3304024, -42.9114227, 45.9673805
3: -15.7773991, 37.6067963, -14.1786652, 32.9450378, -48.7224350, 51.7854614
4: -16.7466125, 35.8476906, -14.6384068, 32.4011078, -49.1477165, 50.4860992

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.1871862, 26.6099091, -33.2752991, 32.0736694
1: -7.7828097, 28.1194229, -8.3826809, 30.0424767, -37.8252754, 36.5021057
2: -8.2539454, 28.2906685, -8.9014492, 30.2400932, -38.4940376, 37.1921158
3: -12.2706099, 28.8829918, -13.1529007, 30.8414726, -43.1120834, 42.0358925
4: -12.8631258, 28.1254616, -13.6967449, 30.2116623, -43.0747833, 41.8222046

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.1840873, 26.5985279, -35.7101097, 41.0966263
1: -10.7727814, 38.5158424, -8.3789291, 30.0294685, -40.8022423, 46.8947716
2: -11.2928276, 38.6079216, -8.8976927, 30.2270870, -41.5199127, 47.5056076
3: -16.8676147, 39.8401985, -13.1470404, 30.8277702, -47.6953850, 52.9872398
4: -17.7444897, 38.1677361, -13.6906538, 30.1988678, -47.9433594, 51.8583908

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.7295799, 28.3989639, -35.0643578, 32.6160660
1: -7.7828097, 28.1194229, -9.0717134, 32.0508652, -39.8336678, 37.1911354
2: -8.2539454, 28.2906685, -9.5743036, 32.3409538, -40.5948982, 37.8649712
3: -12.2706099, 28.8829918, -14.1834040, 32.9562950, -45.2269058, 43.0663910
4: -12.8631258, 28.1254616, -14.6434193, 32.4114075, -45.2745285, 42.7688828

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.7270870, 28.3896866, -37.5012741, 41.6396255
1: -10.7727814, 38.5158424, -9.0686817, 32.0401726, -42.8129539, 47.5845261
2: -11.2928276, 38.6079216, -9.5712700, 32.3304024, -43.6232300, 48.1791916
3: -16.8676147, 39.8401985, -14.1786652, 32.9450378, -49.8126526, 54.0188637
4: -17.7444897, 38.1677361, -14.6384068, 32.4011078, -50.1455956, 52.8061447

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.3701210, 27.3269844, -33.4585876, 30.5112972
1: -7.1131840, 26.1596413, -8.6132603, 30.8662567, -37.9794388, 34.7728996
2: -7.5950131, 26.2545033, -9.1305141, 31.0644073, -38.6594200, 35.3850136
3: -11.2687273, 26.8291836, -13.5239582, 31.7180500, -42.9867783, 40.3531380
4: -11.9468098, 25.9775772, -14.0812511, 31.0096245, -42.9564323, 40.0588188

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.3670039, 27.3154736, -35.8482895, 39.3869896
1: -10.0470028, 36.3811836, -8.6094589, 30.8530769, -40.9000778, 44.9906425
2: -10.5810204, 36.3961143, -9.1267338, 31.0512180, -41.6322365, 45.5228462
3: -15.7773991, 37.6067963, -13.5180254, 31.7041950, -47.4815941, 51.1248207
4: -16.7466125, 35.8476906, -14.0751152, 30.9966831, -47.7432938, 49.9228058

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.9041195, 29.0763073, -35.2079163, 31.0452976
1: -7.1131840, 26.1596413, -9.2882051, 32.8371925, -39.9503708, 35.4478455
2: -7.5950131, 26.2545033, -9.7911968, 33.1104507, -40.7054634, 36.0457001
3: -11.2687273, 26.8291836, -14.5323257, 33.7886276, -45.0573540, 41.3615036
4: -11.9468098, 25.9775772, -15.0085173, 33.1586647, -45.1054764, 40.9860878

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.9010825, 29.0650730, -37.5978889, 39.9210663
1: -10.0470028, 36.3811836, -9.2845058, 32.8242683, -42.8712692, 45.6656876
2: -10.5810204, 36.3961143, -9.7874994, 33.0975723, -43.6785927, 46.1836052
3: -15.7773991, 37.6067963, -14.5265522, 33.7750397, -49.5524368, 52.1333466
4: -16.7466125, 35.8476906, -15.0024967, 33.1459808, -49.8925934, 50.8501892

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.3701210, 27.3269844, -33.9923782, 32.2566032
1: -7.7828097, 28.1194229, -8.6132603, 30.8662567, -38.6490517, 36.7326813
2: -8.2539454, 28.2906685, -9.1305141, 31.0644073, -39.3183479, 37.4211807
3: -12.2706099, 28.8829918, -13.5239582, 31.7180500, -43.9886589, 42.4069519
4: -12.8631258, 28.1254616, -14.0812511, 31.0096245, -43.8727493, 42.2067108

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.3670039, 27.3154736, -36.4270630, 41.2795448
1: -10.7727814, 38.5158424, -8.6094589, 30.8530769, -41.6258583, 47.1253014
2: -11.2928276, 38.6079216, -9.1267338, 31.0512180, -42.3440475, 47.7346573
3: -16.8676147, 39.8401985, -13.5180254, 31.7041950, -48.5718079, 53.3582230
4: -17.7444897, 38.1677361, -14.0751152, 30.9966831, -48.7411728, 52.2428513

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.9041195, 29.0763073, -35.7416992, 32.7906036
1: -7.7828097, 28.1194229, -9.2882051, 32.8371925, -40.6199875, 37.4076271
2: -8.2539454, 28.2906685, -9.7911968, 33.1104507, -41.3643951, 38.0818634
3: -12.2706099, 28.8829918, -14.5323257, 33.7886276, -46.0592384, 43.4153175
4: -12.8631258, 28.1254616, -15.0085173, 33.1586647, -46.0217896, 43.1339760

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.9010825, 29.0650730, -38.1766586, 41.8136215
1: -10.7727814, 38.5158424, -9.2845058, 32.8242683, -43.5970497, 47.8003464
2: -11.2928276, 38.6079216, -9.7874994, 33.0975723, -44.3903999, 48.3954163
3: -16.8676147, 39.8401985, -14.5265522, 33.7750397, -50.6426544, 54.3667526
4: -17.7444897, 38.1677361, -15.0024967, 33.1459808, -50.8904724, 53.1702347

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5691259, upper bound: 43.5588638
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5672675, upper bound: 43.5546301
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5741566, upper bound: 43.5261061
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5653596, upper bound: 43.5237682
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5623864, upper bound: 43.5188589
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5623864
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -6.7334614, 25.0752354, -31.8086967, 31.8086967
1: -7.8301010, 28.3211842, -7.8301010, 28.3211842, -36.1512833, 36.1512833
2: -8.3398962, 28.4661274, -8.3398962, 28.4661274, -36.8060226, 36.8060226
3: -12.3204012, 29.0548534, -12.3204012, 29.0548534, -41.3752556, 41.3752556
4: -12.9191322, 28.3510303, -12.9191322, 28.3510303, -41.2701645, 41.2701645

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5949951, upper bound: 43.5825477
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5819614, upper bound: 43.5792965
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.3271179, 27.2405643, -33.9740257, 32.4023514
1: -7.8301010, 28.3211842, -8.5321484, 30.8015041, -38.6316071, 36.8533325
2: -8.3398962, 28.4661274, -9.0758047, 30.9408817, -39.2807770, 37.5419312
3: -12.3204012, 29.0548534, -13.4046860, 31.6406326, -43.9610329, 42.4595413
4: -12.9191322, 28.3510303, -14.0712585, 30.7746468, -43.6937752, 42.4222794

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5949951, upper bound: 43.5825477
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5819614, upper bound: 43.5792965
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.7334614, 25.0752354, -32.4023514, 33.9740257
1: -8.5321484, 30.8015041, -7.8301010, 28.3211842, -36.8533325, 38.6316071
2: -9.0758047, 30.9408817, -8.3398962, 28.4661274, -37.5419312, 39.2807770
3: -13.4046860, 31.6406326, -12.3204012, 29.0548534, -42.4595413, 43.9610329
4: -14.0712585, 30.7746468, -12.9191322, 28.3510303, -42.4222794, 43.6937752

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5743976, upper bound: 43.5279582
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5233502, upper bound: 43.5233502
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.3271179, 27.2405643, -34.5676727, 34.5676727
1: -8.5321484, 30.8015041, -8.5321484, 30.8015041, -39.3336525, 39.3336525
2: -9.0758047, 30.9408817, -9.0758047, 30.9408817, -40.0166855, 40.0166855
3: -13.4046860, 31.6406326, -13.4046860, 31.6406326, -45.0453186, 45.0453186
4: -14.0712585, 30.7746468, -14.0712585, 30.7746468, -44.8458900, 44.8458900

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5743976, upper bound: 43.5286222
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5251243, upper bound: 43.5241632
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.2693892, 26.8420372, -33.5755005, 32.3446236
1: -7.8301010, 28.3211842, -8.5089540, 30.3026695, -38.1327705, 36.8301392
2: -8.3398962, 28.4661274, -9.0044117, 30.5334988, -38.8733940, 37.4705391
3: -12.3204012, 29.0548534, -13.3364210, 31.1392040, -43.4596024, 42.3912735
4: -12.9191322, 28.3510303, -13.8503227, 30.5232334, -43.4423637, 42.2013550

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5807983, upper bound: 43.5614882
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.8313341, 28.8915958, -35.6250572, 32.9065704
1: -7.8301010, 28.3211842, -9.1737223, 32.6585579, -40.4886589, 37.4949036
2: -8.3398962, 28.4661274, -9.7022839, 32.8745689, -41.2144661, 38.1684113
3: -12.3204012, 29.0548534, -14.3621759, 33.5927696, -45.9131699, 43.4170265
4: -12.9191322, 28.3510303, -14.9443817, 32.8122597, -45.7313843, 43.2954102

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5807983, upper bound: 43.5614882
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.5063033, 24.2676392, -31.5947571, 33.7468681
1: -8.5321484, 30.8015041, -7.5842962, 27.4015598, -35.9337082, 38.3857994
2: -9.0758047, 30.9408817, -8.0552320, 27.5818462, -36.6576462, 38.9961128
3: -13.4046860, 31.6406326, -11.9514847, 28.1189060, -41.5235901, 43.5921173
4: -14.0712585, 30.7746468, -12.5255804, 27.4433079, -41.5145531, 43.3002243

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3242836, 27.2302094, -8.9248209, 33.1853523, -40.5096359, 36.1550293
1: -8.5287476, 30.7896252, -10.5382071, 37.6687279, -46.1974754, 41.3278275
2: -9.0723677, 30.9289951, -11.0602732, 37.7767830, -46.8491516, 41.9892693
3: -13.3993721, 31.6281586, -16.4940701, 38.9501419, -52.3495102, 48.1222267
4: -14.0656605, 30.7630711, -17.3545589, 37.3720055, -51.4376678, 48.1176300

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -5.9647384, 22.4902878, -28.9965916, 30.2323761
1: -7.5842962, 27.4015598, -6.9048386, 25.4210529, -33.0053444, 34.3063965
2: -8.0552320, 27.5818462, -7.3848820, 25.5047226, -33.5599556, 34.9667244
3: -11.9514847, 28.1189060, -10.9337454, 26.0330429, -37.9845276, 39.0526505
4: -12.5255804, 27.4433079, -11.5946884, 25.2559605, -37.7815361, 39.0379829

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589390, upper bound: 43.5691566
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588638, upper bound: 43.5691259
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.3492012, 31.2999210, -37.8062248, 32.6168404
1: -7.5842962, 27.4015598, -9.8115788, 35.5474586, -43.1317520, 37.2131348
2: -8.0552320, 27.5818462, -10.3486872, 35.5690842, -43.6243172, 37.9305344
3: -11.9514847, 28.1189060, -15.4028683, 36.7234039, -48.6748848, 43.5217743
4: -12.5255804, 27.4433079, -16.3597260, 35.0432816, -47.5688629, 43.8030281

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589390, upper bound: 43.5691566
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588638, upper bound: 43.5691259
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -5.9647384, 22.4902878, -31.4151058, 39.1500854
1: -10.5382071, 37.6687279, -6.9048386, 25.4210529, -35.9592590, 44.5735664
2: -11.0602732, 37.7767830, -7.3848820, 25.5047226, -36.5649910, 45.1616669
3: -16.4940701, 38.9501419, -10.9337454, 26.0330429, -42.5271149, 49.8838844
4: -17.3545589, 37.3720055, -11.5946884, 25.2559605, -42.6105156, 48.9666862

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5265996
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5228568
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.3492012, 31.2999210, -40.2247429, 41.5345459
1: -10.5382071, 37.6687279, -9.8115788, 35.5474586, -46.0856628, 47.4803085
2: -11.0602732, 37.7767830, -10.3486872, 35.5690842, -46.6293564, 48.1254692
3: -16.4940701, 38.9501419, -15.4028683, 36.7234039, -53.2174759, 54.3530083
4: -17.3545589, 37.3720055, -16.3597260, 35.0432816, -52.3978424, 53.7317276

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5265996
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5228568
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.5063033, 24.2676392, -30.7739429, 30.7739429
1: -7.5842962, 27.4015598, -7.5842962, 27.4015598, -34.9858475, 34.9858475
2: -8.0552320, 27.5818462, -8.0552320, 27.5818462, -35.6370773, 35.6370773
3: -11.9514847, 28.1189060, -11.9514847, 28.1189060, -40.0703888, 40.0703888
4: -12.5255804, 27.4433079, -12.5255804, 27.4433079, -39.9688873, 39.9688873

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585932, upper bound: 43.5579472
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585180, upper bound: 43.5579165
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.9248209, 33.1853523, -39.6916504, 33.1924591
1: -7.5842962, 27.4015598, -10.5382071, 37.6687279, -45.2530251, 37.9397621
2: -8.0552320, 27.5818462, -11.0602732, 37.7767830, -45.8320160, 38.6421165
3: -11.9514847, 28.1189060, -16.4940701, 38.9501419, -50.9016266, 44.6129761
4: -12.5255804, 27.4433079, -17.3545589, 37.3720055, -49.8975830, 44.7978668

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585932, upper bound: 43.5579472
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585180, upper bound: 43.5579165
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.5063033, 24.2676392, -33.1924591, 39.6916504
1: -10.5382071, 37.6687279, -7.5842962, 27.4015598, -37.9397621, 45.2530251
2: -11.0602732, 37.7767830, -8.0552320, 27.5818462, -38.6421165, 45.8320160
3: -16.4940701, 38.9501419, -11.9514847, 28.1189060, -44.6129761, 50.9016266
4: -17.3545589, 37.3720055, -12.5255804, 27.4433079, -44.7978630, 49.8975830

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5260811
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5223382
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.9248209, 33.1853523, -42.1101685, 42.1101685
1: -10.5382071, 37.6687279, -10.5382071, 37.6687279, -48.2069359, 48.2069359
2: -11.0602732, 37.7767830, -11.0602732, 37.7767830, -48.8370552, 48.8370552
3: -16.4940701, 38.9501419, -16.4940701, 38.9501419, -55.4442139, 55.4442139
4: -17.3545589, 37.3720055, -17.3545589, 37.3720055, -54.7265625, 54.7265625

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5260811
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5223382
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -6.9100652, 25.7693195, -32.5027809, 31.9853001
1: -7.8301010, 28.3211842, -8.0518236, 29.1186275, -36.9487305, 36.3730049
2: -8.3398962, 28.4661274, -8.5616179, 29.2663651, -37.6062622, 37.0277443
3: -12.3204012, 29.0548534, -12.6775093, 29.9030113, -42.2234116, 41.7323608
4: -12.9191322, 28.3510303, -13.2914982, 29.1219196, -42.0410385, 41.6425247

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5918522, upper bound: 43.5604914
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5789399, upper bound: 43.5572545
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.5059209, 27.9394646, -34.6729279, 32.5811577
1: -7.8301010, 28.3211842, -8.7573853, 31.6072998, -39.4374008, 37.0785637
2: -8.3398962, 28.4661274, -9.2995558, 31.7472954, -40.0871925, 37.7656822
3: -12.3204012, 29.0548534, -13.7668066, 32.4964180, -44.8168182, 42.8216591
4: -12.9191322, 28.3510303, -14.4473610, 31.5514050, -44.4705353, 42.7983932

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5918522, upper bound: 43.5604914
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5789399, upper bound: 43.5572545
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.1316075, 23.1411781, -30.4682961, 33.3721657
1: -8.5321484, 30.8015041, -7.1131840, 26.1596413, -34.6917877, 37.9146881
2: -9.0758047, 30.9408817, -7.5950131, 26.2545033, -35.3303032, 38.5358963
3: -13.4046860, 31.6406326, -11.2687273, 26.8291836, -40.2338676, 42.9093590
4: -14.0712585, 30.7746468, -11.9468098, 25.9775772, -40.0488205, 42.7214546

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3242836, 27.2302094, -8.5328178, 32.0199852, -39.3442688, 35.7630272
1: -8.5287476, 30.7896252, -10.0470028, 36.3811836, -44.9099312, 40.8366280
2: -9.0723677, 30.9289951, -10.5810204, 36.3961143, -45.4684830, 41.5100136
3: -13.3993721, 31.6281586, -15.7773991, 37.6067963, -51.0061684, 47.4055557
4: -14.0656605, 30.7630711, -16.7466125, 35.8476906, -49.9133530, 47.5096817

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.4370813, 27.4980087, -34.2314682, 32.5123177
1: -7.8301010, 28.3211842, -8.7175827, 31.0629864, -38.8930893, 37.0387650
2: -8.3398962, 28.4661274, -9.2134495, 31.2856255, -39.6255226, 37.6795769
3: -12.3204012, 29.0548534, -13.6725578, 31.9439163, -44.2643166, 42.7274055
4: -12.9191322, 28.3510303, -14.2024050, 31.2481651, -44.1672935, 42.5534363

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5876652, upper bound: 43.5290275
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5767519, upper bound: 43.5263867
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -8.0359106, 29.6763287, -36.4097900, 33.1111450
1: -7.8301010, 28.3211842, -9.4270086, 33.5660286, -41.3961296, 37.7481918
2: -8.3398962, 28.4661274, -9.9561176, 33.7784920, -42.1183891, 38.4222450
3: -12.3204012, 29.0548534, -14.7677183, 34.5513916, -46.8717918, 43.8225708
4: -12.9191322, 28.3510303, -15.3659382, 33.6877899, -46.6069221, 43.7169685

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5876652, upper bound: 43.5290275
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5767519, upper bound: 43.5263867
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.6653934, 24.8864861, -32.2135963, 33.9059563
1: -8.5321484, 30.8015041, -7.7828097, 28.1194229, -36.6515732, 38.5843086
2: -9.0758047, 30.9408817, -8.2539454, 28.2906685, -37.3664703, 39.1948280
3: -13.4046860, 31.6406326, -12.2706099, 28.8829918, -42.2876778, 43.9112396
4: -14.0712585, 30.7746468, -12.8631258, 28.1254616, -42.1967125, 43.6377678

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5623864, upper bound: 43.5188589
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5623864, upper bound: 43.5188589
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3242836, 27.2302094, -9.1115875, 33.9125404, -41.2368240, 36.3417969
1: -8.5287476, 30.7896252, -10.7727814, 38.5158424, -47.0445900, 41.5624046
2: -9.0723677, 30.9289951, -11.2928276, 38.6079216, -47.6802902, 42.2218246
3: -13.3993721, 31.6281586, -16.8676147, 39.8401985, -53.2395706, 48.4957695
4: -14.0656605, 30.7630711, -17.7444897, 38.1677361, -52.2333984, 48.5075607

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5623864, upper bound: 43.5188589
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5623864, upper bound: 43.5188589
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.1316075, 23.1411781, -29.6474819, 30.3992424
1: -7.5842962, 27.4015598, -7.1131840, 26.1596413, -33.7439346, 34.5147400
2: -8.0552320, 27.5818462, -7.5950131, 26.2545033, -34.3097343, 35.1768570
3: -11.9514847, 28.1189060, -11.2687273, 26.8291836, -38.7806625, 39.3876343
4: -12.5255804, 27.4433079, -11.9468098, 25.9775772, -38.5031548, 39.3901176

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555713, upper bound: 43.5453406
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5554961, upper bound: 43.5453099
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.5328178, 32.0199852, -38.5262871, 32.8004570
1: -7.5842962, 27.4015598, -10.0470028, 36.3811836, -43.9654770, 37.4485626
2: -8.0552320, 27.5818462, -10.5810204, 36.3961143, -44.4513474, 38.1628647
3: -11.9514847, 28.1189060, -15.7773991, 37.6067963, -49.5582809, 43.8963051
4: -12.5255804, 27.4433079, -16.7466125, 35.8476906, -48.3732719, 44.1899185

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555713, upper bound: 43.5453406
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5554961, upper bound: 43.5453099
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.1316075, 23.1411781, -32.0659981, 39.3169518
1: -10.5382071, 37.6687279, -7.1131840, 26.1596413, -36.6978455, 44.7819138
2: -11.0602732, 37.7767830, -7.5950131, 26.2545033, -37.3147774, 45.3717957
3: -16.4940701, 38.9501419, -11.2687273, 26.8291836, -43.3232498, 50.2188683
4: -17.3545589, 37.3720055, -11.9468098, 25.9775772, -43.3321342, 49.3188133

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5523180, upper bound: 43.5263945
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218139, upper bound: 43.5226517
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.5328178, 32.0199852, -40.9448051, 41.7181664
1: -10.5382071, 37.6687279, -10.0470028, 36.3811836, -46.9193916, 47.7157288
2: -11.0602732, 37.7767830, -10.5810204, 36.3961143, -47.4563866, 48.3578033
3: -16.4940701, 38.9501419, -15.7773991, 37.6067963, -54.1008682, 54.7275391
4: -17.3545589, 37.3720055, -16.7466125, 35.8476906, -53.2022476, 54.1186142

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5523180, upper bound: 43.5263945
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218139, upper bound: 43.5226517
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.6653934, 24.8864861, -31.3927898, 30.9330311
1: -7.5842962, 27.4015598, -7.7828097, 28.1194229, -35.7037201, 35.1843605
2: -8.0552320, 27.5818462, -8.2539454, 28.2906685, -36.3459015, 35.8357849
3: -11.9514847, 28.1189060, -12.2706099, 28.8829918, -40.8344765, 40.3895149
4: -12.5255804, 27.4433079, -12.8631258, 28.1254616, -40.6510429, 40.3064346

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535277, upper bound: 43.5200645
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5534525, upper bound: 43.5200338
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -9.1115875, 33.9125404, -40.4188423, 33.3792267
1: -7.5842962, 27.4015598, -10.7727814, 38.5158424, -46.1001358, 38.1743393
2: -8.0552320, 27.5818462, -11.2928276, 38.6079216, -46.6631546, 38.8746719
3: -11.9514847, 28.1189060, -16.8676147, 39.8401985, -51.7916832, 44.9865189
4: -12.5255804, 27.4433079, -17.7444897, 38.1677361, -50.6933174, 45.1877975

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535277, upper bound: 43.5200645
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5534525, upper bound: 43.5200338
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.6653934, 24.8864861, -33.8113060, 39.8507423
1: -10.5382071, 37.6687279, -7.7828097, 28.1194229, -38.6576309, 45.4515343
2: -11.0602732, 37.7767830, -8.2539454, 28.2906685, -39.3509407, 46.0307274
3: -16.4940701, 38.9501419, -12.2706099, 28.8829918, -45.3770599, 51.2207527
4: -17.3545589, 37.3720055, -12.8631258, 28.1254616, -45.4800186, 50.2351265

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210155, upper bound: 43.5147703
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -9.1115875, 33.9125404, -42.8373604, 42.2969398
1: -10.5382071, 37.6687279, -10.7727814, 38.5158424, -49.0540466, 48.4415092
2: -11.0602732, 37.7767830, -11.2928276, 38.6079216, -49.6681938, 49.0696106
3: -16.4940701, 38.9501419, -16.8676147, 39.8401985, -56.3342667, 55.8177567
4: -17.3545589, 37.3720055, -17.7444897, 38.1677361, -55.5222931, 55.1164932

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210155, upper bound: 43.5147703
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -5.9647384, 22.4902878, -28.6218910, 29.1059151
1: -7.1131840, 26.1596413, -6.9048386, 25.4210529, -32.5342369, 33.0644798
2: -7.5950131, 26.2545033, -7.3848820, 25.5047226, -33.0997314, 33.6393814
3: -11.2687273, 26.8291836, -10.9337454, 26.0330429, -37.3017693, 37.7629204
4: -11.9468098, 25.9775772, -11.5946884, 25.2559605, -37.2027702, 37.5722542

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5723572, upper bound: 43.5714867
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546301, upper bound: 43.5672675
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -8.3492012, 31.2999210, -37.4315300, 31.4903793
1: -7.1131840, 26.1596413, -9.8115788, 35.5474586, -42.6606445, 35.9712219
2: -7.5950131, 26.2545033, -10.3486872, 35.5690842, -43.1640968, 36.6031914
3: -11.2687273, 26.8291836, -15.4028683, 36.7234039, -47.9921303, 42.2320442
4: -11.9468098, 25.9775772, -16.3597260, 35.0432816, -46.9900894, 42.3372993

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5723572, upper bound: 43.5714867
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546301, upper bound: 43.5672675
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -5.9647384, 22.4902878, -31.0231018, 37.9847221
1: -10.0470028, 36.3811836, -6.9048386, 25.4210529, -35.4680557, 43.2860222
2: -10.5810204, 36.3961143, -7.3848820, 25.5047226, -36.0857315, 43.7809944
3: -15.7773991, 37.6067963, -10.9337454, 26.0330429, -41.8104401, 48.5405426
4: -16.7466125, 35.8476906, -11.5946884, 25.2559605, -42.0025711, 47.4423752

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5235777
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5223325
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -8.3492012, 31.2999210, -39.8327370, 40.3691864
1: -10.0470028, 36.3811836, -9.8115788, 35.5474586, -45.5944595, 46.1927643
2: -10.5810204, 36.3961143, -10.3486872, 35.5690842, -46.1501045, 46.7448006
3: -15.7773991, 37.6067963, -15.4028683, 36.7234039, -52.5008011, 53.0096664
4: -16.7466125, 35.8476906, -16.3597260, 35.0432816, -51.7898941, 52.2074165

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5235777
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5223325
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -6.5063033, 24.2676392, -30.3992424, 29.6474819
1: -7.1131840, 26.1596413, -7.5842962, 27.4015598, -34.5147400, 33.7439308
2: -7.5950131, 26.2545033, -8.0552320, 27.5818462, -35.1768608, 34.3097343
3: -11.2687273, 26.8291836, -11.9514847, 28.1189060, -39.3876343, 38.7806625
4: -11.9468098, 25.9775772, -12.5255804, 27.4433079, -39.3901138, 38.5031509

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5720115, upper bound: 43.5602780
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542844, upper bound: 43.5560581
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -8.9248209, 33.1853523, -39.3169556, 32.0659981
1: -7.1131840, 26.1596413, -10.5382071, 37.6687279, -44.7819138, 36.6978455
2: -7.5950131, 26.2545033, -11.0602732, 37.7767830, -45.3717957, 37.3147774
3: -11.2687273, 26.8291836, -16.4940701, 38.9501419, -50.2188683, 43.3232498
4: -11.9468098, 25.9775772, -17.3545589, 37.3720055, -49.3188133, 43.3321304

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5720115, upper bound: 43.5602780
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542844, upper bound: 43.5560581
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -6.5063033, 24.2676392, -32.8004570, 38.5262871
1: -10.0470028, 36.3811836, -7.5842962, 27.4015598, -37.4485626, 43.9654808
2: -10.5810204, 36.3961143, -8.0552320, 27.5818462, -38.1628647, 44.4513474
3: -15.7773991, 37.6067963, -11.9514847, 28.1189060, -43.8963051, 49.5582809
4: -16.7466125, 35.8476906, -12.5255804, 27.4433079, -44.1899185, 48.3732719

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5400465, upper bound: 43.5230592
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5226517, upper bound: 43.5218139
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -8.9248209, 33.1853523, -41.7181664, 40.9448051
1: -10.0470028, 36.3811836, -10.5382071, 37.6687279, -47.7157288, 46.9193916
2: -10.5810204, 36.3961143, -11.0602732, 37.7767830, -48.3578033, 47.4563866
3: -15.7773991, 37.6067963, -16.4940701, 38.9501419, -54.7275391, 54.1008682
4: -16.7466125, 35.8476906, -17.3545589, 37.3720055, -54.1186142, 53.2022476

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5400465, upper bound: 43.5230592
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5226517, upper bound: 43.5218139
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -5.9647384, 22.4902878, -29.1556797, 30.8512211
1: -7.7828097, 28.1194229, -6.9048386, 25.4210529, -33.2038612, 35.0242615
2: -8.2539454, 28.2906685, -7.3848820, 25.5047226, -33.7586594, 35.6755524
3: -12.2706099, 28.8829918, -10.9337454, 26.0330429, -38.3036537, 39.8167343
4: -12.8631258, 28.1254616, -11.5946884, 25.2559605, -38.1190834, 39.7201462

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5262618, upper bound: 43.5667738
time: 1.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237682, upper bound: 43.5653596
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -8.3492012, 31.2999210, -37.9653130, 33.2356834
1: -7.7828097, 28.1194229, -9.8115788, 35.5474586, -43.3302612, 37.9309998
2: -8.2539454, 28.2906685, -10.3486872, 35.5690842, -43.8230286, 38.6393547
3: -12.2706099, 28.8829918, -15.4028683, 36.7234039, -48.9940109, 44.2858582
4: -12.8631258, 28.1254616, -16.3597260, 35.0432816, -47.9064064, 44.4851875

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5262618, upper bound: 43.5667738
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237682, upper bound: 43.5653596
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -5.9647384, 22.4902878, -31.6018753, 39.8772774
1: -10.7727814, 38.5158424, -6.9048386, 25.4210529, -36.1938324, 45.4206810
2: -11.2928276, 38.6079216, -7.3848820, 25.5047226, -36.7975464, 45.9928055
3: -16.8676147, 39.8401985, -10.9337454, 26.0330429, -42.9006577, 50.7739410
4: -17.7444897, 38.1677361, -11.5946884, 25.2559605, -43.0004463, 49.7624207

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 46

Time for candidate selection: 4.24 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5174363, upper bound: 43.5566163
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161923, upper bound: 43.5556373
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -8.3492012, 31.2999210, -40.4115067, 42.2617416
1: -10.7727814, 38.5158424, -9.8115788, 35.5474586, -46.3202400, 48.3274231
2: -11.2928276, 38.6079216, -10.3486872, 35.5690842, -46.8619118, 48.9566078
3: -16.8676147, 39.8401985, -15.4028683, 36.7234039, -53.5910187, 55.2430649
4: -17.7444897, 38.1677361, -16.3597260, 35.0432816, -52.7877731, 54.5274620

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 46

Time for candidate selection: 4.00 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5174363, upper bound: 43.5566163
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161923, upper bound: 43.5561738
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -6.5063033, 24.2676392, -30.9330311, 31.3927879
1: -7.7828097, 28.1194229, -7.5842962, 27.4015598, -35.1843605, 35.7037201
2: -8.2539454, 28.2906685, -8.0552320, 27.5818462, -35.8357849, 36.3459015
3: -12.2706099, 28.8829918, -11.9514847, 28.1189060, -40.3895111, 40.8344765
4: -12.8631258, 28.1254616, -12.5255804, 27.4433079, -40.3064308, 40.6510429

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259161, upper bound: 43.5555644
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5234224, upper bound: 43.5542586
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -8.9248209, 33.1853523, -39.8507385, 33.8113060
1: -7.7828097, 28.1194229, -10.5382071, 37.6687279, -45.4515343, 38.6576309
2: -8.2539454, 28.2906685, -11.0602732, 37.7767830, -46.0307274, 39.3509407
3: -12.2706099, 28.8829918, -16.4940701, 38.9501419, -51.2207527, 45.3770599
4: -12.8631258, 28.1254616, -17.3545589, 37.3720055, -50.2351265, 45.4800186

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259161, upper bound: 43.5555644
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5234224, upper bound: 43.5542586
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -6.5063033, 24.2676392, -33.3792267, 40.4188423
1: -10.7727814, 38.5158424, -7.5842962, 27.4015598, -38.1743393, 46.1001358
2: -11.2928276, 38.6079216, -8.0552320, 27.5818462, -38.8746719, 46.6631546
3: -16.8676147, 39.8401985, -11.9514847, 28.1189060, -44.9865189, 51.7916832
4: -17.7444897, 38.1677361, -12.5255804, 27.4433079, -45.1877975, 50.6933174

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 46

Time for candidate selection: 3.83 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5173365, upper bound: 43.5491488
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5160924, upper bound: 43.5481698
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -8.9248209, 33.1853523, -42.2969360, 42.8373604
1: -10.7727814, 38.5158424, -10.5382071, 37.6687279, -48.4415092, 49.0540466
2: -11.2928276, 38.6079216, -11.0602732, 37.7767830, -49.0696106, 49.6681938
3: -16.8676147, 39.8401985, -16.4940701, 38.9501419, -55.8177567, 56.3342667
4: -17.7444897, 38.1677361, -17.3545589, 37.3720055, -55.1164932, 55.5222931

Time for backsubstitution: 2.21 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5950900, upper bound: 43.5779056
time: 0.47 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 3, lower bound: -43.5950900, upper bound: 43.5779056
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.6606002, 28.3044319, -35.7576599, 35.2311859
1: -8.7061062, 31.1410255, -8.9601765, 31.9744511, -40.6805573, 40.1012039
2: -9.2308626, 31.3365688, -9.4893942, 32.1866379, -41.4174995, 40.8259621
3: -13.6493845, 31.9803791, -14.0368652, 32.8534927, -46.5028763, 46.0172424
4: -14.1980610, 31.2951145, -14.5853920, 32.1437492, -46.3418121, 45.8805046

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.45 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.6094785, 28.1359310, -35.7740211, 35.8971863
1: -8.9388247, 31.9677963, -8.8971281, 31.7846928, -40.7235069, 40.8649254
2: -9.4615984, 32.1662331, -9.4261189, 31.9909630, -41.4525604, 41.5923538
3: -14.0235004, 32.8585358, -13.9439011, 32.6553459, -46.6788483, 46.8024330
4: -14.5844440, 32.0941963, -14.4959812, 31.9436836, -46.5281296, 46.5901756

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.4532280, 27.5705853, -35.0238037, 35.0238037
1: -8.7061062, 31.1410255, -8.7061062, 31.1410255, -39.8471298, 39.8471298
2: -9.2308626, 31.3365688, -9.2308626, 31.3365688, -40.5674324, 40.5674324
3: -13.6493845, 31.9803791, -13.6493845, 31.9803791, -45.6297646, 45.6297646
4: -14.1980610, 31.2951145, -14.1980610, 31.2951145, -45.4931755, 45.4931755

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5766672, upper bound: 43.5300538
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
time: 0.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.6380959, 28.2877064, -35.7409325, 35.2086716
1: -8.7061062, 31.1410255, -8.9388247, 31.9677963, -40.6739006, 40.0798416
2: -9.2308626, 31.3365688, -9.4615984, 32.1662331, -41.3970947, 40.7981682
3: -13.6493845, 31.9803791, -14.0235004, 32.8585358, -46.5079193, 46.0038795
4: -14.1980610, 31.2951145, -14.5844440, 32.0941963, -46.2922554, 45.8795586

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5766672, upper bound: 43.5300538
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
time: 0.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.4532280, 27.5705853, -35.2086716, 35.7409363
1: -8.9388247, 31.9677963, -8.7061062, 31.1410255, -40.0798416, 40.6739044
2: -9.4615984, 32.1662331, -9.2308626, 31.3365688, -40.7981682, 41.3970947
3: -14.0235004, 32.8585358, -13.6493845, 31.9803791, -46.0038795, 46.5079193
4: -14.5844440, 32.0941963, -14.1980610, 31.2951145, -45.8795586, 46.2922554

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5300713
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.6380959, 28.2877064, -35.9258003, 35.9258003
1: -8.9388247, 31.9677963, -8.9388247, 31.9677963, -40.9066124, 40.9066124
2: -9.4615984, 32.1662331, -9.4615984, 32.1662331, -41.6278305, 41.6278305
3: -14.0235004, 32.8585358, -14.0235004, 32.8585358, -46.8820343, 46.8820343
4: -14.5844440, 32.0941963, -14.5844440, 32.0941963, -46.6786385, 46.6786385

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5300713
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.21 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5766672, upper bound: 43.5300538
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5766672, upper bound: 43.5300538
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5300713
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5300713
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.4532280, 27.5705853, -34.7577667, 34.0631332
1: -8.3826809, 30.0424767, -8.7061062, 31.1410255, -39.5237045, 38.7485809
2: -8.9014492, 30.2400932, -9.2308626, 31.3365688, -40.2380180, 39.4709549
3: -13.1529007, 30.8414726, -13.6493845, 31.9803791, -45.1332779, 44.4908562
4: -13.6967449, 30.2116623, -14.1980610, 31.2951145, -44.9918594, 44.4097214

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.4241757, 27.4706249, -35.2002029, 35.8231392
1: -9.0717134, 32.0508652, -8.6702576, 31.0274582, -40.0991592, 40.7211227
2: -9.5743036, 32.3409538, -9.1947222, 31.2212181, -40.7955208, 41.5356750
3: -14.1834040, 32.9562950, -13.5951538, 31.8610687, -46.0444641, 46.5514488
4: -14.6434193, 32.4114075, -14.1446362, 31.1782990, -45.8217163, 46.5560341

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.6380959, 28.2877064, -35.4748917, 34.2480011
1: -8.3826809, 30.0424767, -8.9388247, 31.9677963, -40.3504791, 38.9812927
2: -8.9014492, 30.2400932, -9.4615984, 32.1662331, -41.0676804, 39.7016907
3: -13.1529007, 30.8414726, -14.0235004, 32.8585358, -46.0114365, 44.8649750
4: -13.6967449, 30.2116623, -14.5844440, 32.0941963, -45.7909393, 44.7961044

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.6076822, 28.1827126, -35.9122925, 36.0066452
1: -9.0717134, 32.0508652, -8.9013500, 31.8481598, -40.9198685, 40.9522057
2: -9.5743036, 32.3409538, -9.4238243, 32.0450859, -41.6193886, 41.7647743
3: -14.1834040, 32.9562950, -13.9667282, 32.7329941, -46.9163933, 46.9230194
4: -14.6434193, 32.4114075, -14.5283165, 31.9720306, -46.6154480, 46.9397163

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.4532280, 27.5705853, -34.9407043, 34.7802124
1: -8.6132603, 30.8662567, -8.7061062, 31.1410255, -39.7542839, 39.5723572
2: -9.1305141, 31.0644073, -9.2308626, 31.3365688, -40.4670830, 40.2952690
3: -13.5239582, 31.7180500, -13.6493845, 31.9803791, -45.5043373, 45.3674355
4: -14.0812511, 31.0096245, -14.1980610, 31.2951145, -45.3763618, 45.2076836

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.4241757, 27.4706249, -35.3747444, 36.5004845
1: -9.2882051, 32.8371925, -8.6702576, 31.0274582, -40.3156586, 41.5074463
2: -9.7911968, 33.1104507, -9.1947222, 31.2212181, -41.0124130, 42.3051720
3: -14.5323257, 33.7886276, -13.5951538, 31.8610687, -46.3933907, 47.3837814
4: -15.0085173, 33.1586647, -14.1446362, 31.1782990, -46.1868134, 47.3032951

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.6380959, 28.2877064, -35.6578293, 34.9650764
1: -8.6132603, 30.8662567, -8.9388247, 31.9677963, -40.5810547, 39.8050690
2: -9.1305141, 31.0644073, -9.4615984, 32.1662331, -41.2967453, 40.5260048
3: -13.5239582, 31.7180500, -14.0235004, 32.8585358, -46.3824921, 45.7415504
4: -14.0812511, 31.0096245, -14.5844440, 32.0941963, -46.1754379, 45.5940704

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.6076822, 28.1827126, -36.0868301, 36.6839905
1: -9.2882051, 32.8371925, -8.9013500, 31.8481598, -41.1363640, 41.7385254
2: -9.7911968, 33.1104507, -9.4238243, 32.0450859, -41.8362808, 42.5342751
3: -14.5323257, 33.7886276, -13.9667282, 32.7329941, -47.2653198, 47.7553520
4: -15.0085173, 33.1586647, -14.5283165, 31.9720306, -46.9805412, 47.6869774

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5611918, upper bound: 43.5611918
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5583637, upper bound: 43.5278256
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5278256, upper bound: 43.5583637
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.1871862, 26.6099091, -33.7970886, 33.7970886
1: -8.3826809, 30.0424767, -8.3826809, 30.0424767, -38.4251556, 38.4251556
2: -8.9014492, 30.2400932, -8.9014492, 30.2400932, -39.1415405, 39.1415405
3: -13.1529007, 30.8414726, -13.1529007, 30.8414726, -43.9943733, 43.9943733
4: -13.6967449, 30.2116623, -13.6967449, 30.2116623, -43.9084091, 43.9084091

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5918492, upper bound: 43.5637368
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.7295799, 28.3989639, -35.5861473, 34.3394890
1: -8.3826809, 30.0424767, -9.0717134, 32.0508652, -40.4335480, 39.1141815
2: -8.9014492, 30.2400932, -9.5743036, 32.3409538, -41.2424011, 39.8143959
3: -13.1529007, 30.8414726, -14.1834040, 32.9562950, -46.1091957, 45.0248756
4: -13.6967449, 30.2116623, -14.6434193, 32.4114075, -46.1081543, 44.8550797

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5918492, upper bound: 43.5637368
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.1871862, 26.6099091, -34.3394890, 35.5861473
1: -9.0717134, 32.0508652, -8.3826809, 30.0424767, -39.1141815, 40.4335480
2: -9.5743036, 32.3409538, -8.9014492, 30.2400932, -39.8143959, 41.2424011
3: -14.1834040, 32.9562950, -13.1529007, 30.8414726, -45.0248756, 46.1091957
4: -14.6434193, 32.4114075, -13.6967449, 30.2116623, -44.8550797, 46.1081543

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.7295799, 28.3989639, -36.1285439, 36.1285439
1: -9.0717134, 32.0508652, -9.0717134, 32.0508652, -41.1225739, 41.1225739
2: -9.5743036, 32.3409538, -9.5743036, 32.3409538, -41.9152565, 41.9152565
3: -14.1834040, 32.9562950, -14.1834040, 32.9562950, -47.1396904, 47.1396904
4: -14.6434193, 32.4114075, -14.6434193, 32.4114075, -47.0548248, 47.0548248

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.3701210, 27.3269844, -34.5141716, 33.9800301
1: -8.3826809, 30.0424767, -8.6132603, 30.8662567, -39.2489357, 38.6557350
2: -8.9014492, 30.2400932, -9.1305141, 31.0644073, -39.9658546, 39.3706055
3: -13.1529007, 30.8414726, -13.5239582, 31.7180500, -44.8709488, 44.3654327
4: -13.6967449, 30.2116623, -14.0812511, 31.0096245, -44.7063675, 44.2929077

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5663383, upper bound: 43.5264556
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.9041195, 29.0763073, -36.2634926, 34.5140305
1: -8.3826809, 30.0424767, -9.2882051, 32.8371925, -41.2198677, 39.3306808
2: -8.9014492, 30.2400932, -9.7911968, 33.1104507, -42.0119019, 40.0312881
3: -13.1529007, 30.8414726, -14.5323257, 33.7886276, -46.9415283, 45.3737984
4: -13.6967449, 30.2116623, -15.0085173, 33.1586647, -46.8554077, 45.2201729

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5663383, upper bound: 43.5264556
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.3701210, 27.3269844, -35.0565643, 35.7690849
1: -9.0717134, 32.0508652, -8.6132603, 30.8662567, -39.9379578, 40.6641235
2: -9.5743036, 32.3409538, -9.1305141, 31.0644073, -40.6387062, 41.4714661
3: -14.1834040, 32.9562950, -13.5239582, 31.7180500, -45.9014473, 46.4802551
4: -14.6434193, 32.4114075, -14.0812511, 31.0096245, -45.6530418, 46.4926529

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5485978, upper bound: 43.5195879
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.9041195, 29.0763073, -36.8058853, 36.3030815
1: -9.0717134, 32.0508652, -9.2882051, 32.8371925, -41.9088936, 41.3390694
2: -9.5743036, 32.3409538, -9.7911968, 33.1104507, -42.6847534, 42.1321487
3: -14.1834040, 32.9562950, -14.5323257, 33.7886276, -47.9720230, 47.4886169
4: -14.6434193, 32.4114075, -15.0085173, 33.1586647, -47.8020859, 47.4199181

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5485978, upper bound: 43.5195879
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.1871862, 26.6099091, -33.9800262, 34.5141678
1: -8.6132603, 30.8662567, -8.3826809, 30.0424767, -38.6557350, 39.2489357
2: -9.1305141, 31.0644073, -8.9014492, 30.2400932, -39.3706055, 39.9658546
3: -13.5239582, 31.7180500, -13.1529007, 30.8414726, -44.3654327, 44.8709488
4: -14.0812511, 31.0096245, -13.6967449, 30.2116623, -44.2929077, 44.7063675

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.7295799, 28.3989639, -35.7690849, 35.0565643
1: -8.6132603, 30.8662567, -9.0717134, 32.0508652, -40.6641235, 39.9379578
2: -9.1305141, 31.0644073, -9.5743036, 32.3409538, -41.4714661, 40.6387062
3: -13.5239582, 31.7180500, -14.1834040, 32.9562950, -46.4802551, 45.9014473
4: -14.0812511, 31.0096245, -14.6434193, 32.4114075, -46.4926529, 45.6530418

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.1871862, 26.6099091, -34.5140305, 36.2634926
1: -9.2882051, 32.8371925, -8.3826809, 30.0424767, -39.3306808, 41.2198677
2: -9.7911968, 33.1104507, -8.9014492, 30.2400932, -40.0312881, 42.0119019
3: -14.5323257, 33.7886276, -13.1529007, 30.8414726, -45.3737984, 46.9415283
4: -15.0085173, 33.1586647, -13.6967449, 30.2116623, -45.2201729, 46.8554077

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259964, upper bound: 43.5547796
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.7295799, 28.3989639, -36.3030853, 36.8058853
1: -9.2882051, 32.8371925, -9.0717134, 32.0508652, -41.3390694, 41.9088936
2: -9.7911968, 33.1104507, -9.5743036, 32.3409538, -42.1321487, 42.6847534
3: -14.5323257, 33.7886276, -14.1834040, 32.9562950, -47.4886169, 47.9720230
4: -15.0085173, 33.1586647, -14.6434193, 32.4114075, -47.4199181, 47.8020859

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259964, upper bound: 43.5547796
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.3701210, 27.3269844, -34.6971054, 34.6971054
1: -8.6132603, 30.8662567, -8.6132603, 30.8662567, -39.4795113, 39.4795113
2: -9.1305141, 31.0644073, -9.1305141, 31.0644073, -40.1949158, 40.1949196
3: -13.5239582, 31.7180500, -13.5239582, 31.7180500, -45.2420082, 45.2420082
4: -14.0812511, 31.0096245, -14.0812511, 31.0096245, -45.0908699, 45.0908699

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3701210, 27.3269844, -7.9041195, 29.0763073, -36.4464264, 35.2311020
1: -8.6132603, 30.8662567, -9.2882051, 32.8371925, -41.4504471, 40.1544571
2: -9.1305141, 31.0644073, -9.7911968, 33.1104507, -42.2409668, 40.8556061
3: -13.5239582, 31.7180500, -14.5323257, 33.7886276, -47.3125839, 46.2503738
4: -14.0812511, 31.0096245, -15.0085173, 33.1586647, -47.2399139, 46.0181351

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.3701210, 27.3269844, -35.2311020, 36.4464264
1: -9.2882051, 32.8371925, -8.6132603, 30.8662567, -40.1544571, 41.4504471
2: -9.7911968, 33.1104507, -9.1305141, 31.0644073, -40.8556061, 42.2409668
3: -14.5323257, 33.7886276, -13.5239582, 31.7180500, -46.2503738, 47.3125839
4: -15.0085173, 33.1586647, -14.0812511, 31.0096245, -46.0181351, 47.2399139

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213025, upper bound: 43.5173906
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.9041195, 29.0763073, -7.9041195, 29.0763073, -36.9804268, 36.9804268
1: -9.2882051, 32.8371925, -9.2882051, 32.8371925, -42.1253929, 42.1253929
2: -9.7911968, 33.1104507, -9.7911968, 33.1104507, -42.9016495, 42.9016495
3: -14.5323257, 33.7886276, -14.5323257, 33.7886276, -48.3209496, 48.3209496
4: -15.0085173, 33.1586647, -15.0085173, 33.1586647, -48.1671791, 48.1671791

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213025, upper bound: 43.5173906
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 1.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.77 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5918492, upper bound: 43.5637368
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5918492, upper bound: 43.5637368
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5663383, upper bound: 43.5264556
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5663383, upper bound: 43.5264556
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5485978, upper bound: 43.5195879
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5485978, upper bound: 43.5195879
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5259964, upper bound: 43.5547796
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5259964, upper bound: 43.5547796
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5213025, upper bound: 43.5173906
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5213025, upper bound: 43.5173906
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.77
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.1871862, 26.6099091, -33.3433685, 32.2624207
1: -7.8301010, 28.3211842, -8.3826809, 30.0424767, -37.8725777, 36.7038651
2: -8.3398962, 28.4661274, -8.9014492, 30.2400932, -38.5799904, 37.3675766
3: -12.3204012, 29.0548534, -13.1529007, 30.8414726, -43.1618729, 42.2077560
4: -12.9191322, 28.3510303, -13.6967449, 30.2116623, -43.1307907, 42.0477753

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.1645679, 26.5306911, -33.8578033, 34.4051208
1: -8.5321484, 30.8015041, -8.3542242, 29.9541130, -38.4862556, 39.1557274
2: -9.0758047, 30.9408817, -8.8735209, 30.1469250, -39.2227287, 39.8143997
3: -13.4046860, 31.6406326, -13.1095076, 30.7489510, -44.1536369, 44.7501411
4: -14.0712585, 30.7746468, -13.6569252, 30.1153564, -44.1866035, 44.4315720

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5702890, upper bound: 43.5766722
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5679510, upper bound: 43.5679510
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.7295799, 28.3989639, -35.1324234, 32.8048172
1: -7.8301010, 28.3211842, -9.0717134, 32.0508652, -39.8809662, 37.3928909
2: -8.3398962, 28.4661274, -9.5743036, 32.3409538, -40.6808510, 38.0404320
3: -12.3204012, 29.0548534, -14.1834040, 32.9562950, -45.2766953, 43.2382507
4: -12.9191322, 28.3510303, -14.6434193, 32.4114075, -45.3305359, 42.9944496

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.7056627, 28.3152580, -35.6423721, 34.9462242
1: -8.5321484, 30.8015041, -9.0415726, 31.9572430, -40.4893913, 39.8430786
2: -9.0758047, 30.9408817, -9.5447874, 32.2426224, -41.3184204, 40.4856682
3: -13.4046860, 31.6406326, -14.1375294, 32.8581543, -46.2628403, 45.7781601
4: -14.0712585, 30.7746468, -14.6009455, 32.3097420, -46.3809853, 45.3755875

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5691259, upper bound: 43.5588638
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.1871862, 26.6099091, -33.1162109, 31.4548264
1: -7.5842962, 27.4015598, -8.3826809, 30.0424767, -37.6267662, 35.7842407
2: -8.0552320, 27.5818462, -8.9014492, 30.2400932, -38.2953262, 36.4832954
3: -11.9514847, 28.1189060, -13.1529007, 30.8414726, -42.7929573, 41.2718048
4: -12.5255804, 27.4433079, -13.6967449, 30.2116623, -42.7372398, 41.1400528

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.1430955, 26.4489613, -35.3737831, 40.3284416
1: -10.5382071, 37.6687279, -8.3301830, 29.8583546, -40.3965530, 45.9989090
2: -11.0602732, 37.7767830, -8.8481064, 30.0566387, -41.1169128, 46.6248894
3: -16.4940701, 38.9501419, -13.0709028, 30.6479645, -47.1420364, 52.0210457
4: -17.3545589, 37.3720055, -13.6098948, 30.0314522, -47.3860092, 50.9818954

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.7295799, 28.3989639, -34.9052658, 31.9972191
1: -7.5842962, 27.4015598, -9.0717134, 32.0508652, -39.6351585, 36.4732628
2: -8.0552320, 27.5818462, -9.5743036, 32.3409538, -40.3961868, 37.1561470
3: -11.9514847, 28.1189060, -14.1834040, 32.9562950, -44.9077797, 42.3023033
4: -12.5255804, 27.4433079, -14.6434193, 32.4114075, -44.9369850, 42.0867271

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.6920562, 28.2595654, -37.1843872, 40.8774071
1: -10.5382071, 37.6687279, -9.0262203, 31.8900490, -42.4282532, 46.6949463
2: -11.0602732, 37.7767830, -9.5286884, 32.1826820, -43.2429543, 47.3054733
3: -16.4940701, 38.9501419, -14.1123037, 32.7868042, -49.2808762, 53.0624428
4: -17.3545589, 37.3720055, -14.5677023, 32.2570839, -49.6116409, 51.9397049

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.3701210, 27.3269844, -34.0604477, 32.4453583
1: -7.8301010, 28.3211842, -8.6132603, 30.8662567, -38.6963577, 36.9344444
2: -8.3398962, 28.4661274, -9.1305141, 31.0644073, -39.4043045, 37.5966415
3: -12.3204012, 29.0548534, -13.5239582, 31.7180500, -44.0384521, 42.5788116
4: -12.9191322, 28.3510303, -14.0812511, 31.0096245, -43.9287529, 42.4322815

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5749305, upper bound: 43.5568833
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5749305, upper bound: 43.5568833
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.3431458, 27.2304611, -34.5575752, 34.5837097
1: -8.5321484, 30.8015041, -8.5792074, 30.7582092, -39.2903595, 39.3807106
2: -9.0758047, 30.9408817, -9.0971594, 30.9519367, -40.0277405, 40.0380402
3: -13.4046860, 31.6406326, -13.4719820, 31.6049385, -45.0096245, 45.1126137
4: -14.0712585, 30.7746468, -14.0329437, 30.8937836, -44.9650307, 44.8075867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5669360, upper bound: 43.5546301
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5641460, upper bound: 43.5441267
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.9041195, 29.0763073, -35.8097687, 32.9793549
1: -7.8301010, 28.3211842, -9.2882051, 32.8371925, -40.6672935, 37.6093903
2: -8.3398962, 28.4661274, -9.7911968, 33.1104507, -41.4503479, 38.2573242
3: -12.3204012, 29.0548534, -14.5323257, 33.7886276, -46.1090279, 43.5871773
4: -12.9191322, 28.3510303, -15.0085173, 33.1586647, -46.0777969, 43.3595467

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -7.8762832, 28.9767017, -36.3038177, 35.1168404
1: -8.5321484, 30.8015041, -9.2530289, 32.7255402, -41.2576828, 40.0545349
2: -9.0758047, 30.9408817, -9.7568846, 32.9937859, -42.0695877, 40.6977654
3: -13.4046860, 31.6406326, -14.4785843, 33.6718712, -47.0765572, 46.1192169
4: -14.0712585, 30.7746468, -14.9586267, 33.0391541, -47.1103973, 45.7332726

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5532520, upper bound: 43.5230864
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.3701210, 27.3269844, -33.8332863, 31.6377602
1: -7.5842962, 27.4015598, -8.6132603, 30.8662567, -38.4505424, 36.0148125
2: -8.0552320, 27.5818462, -9.1305141, 31.0644073, -39.1196404, 36.7123566
3: -11.9514847, 28.1189060, -13.5239582, 31.7180500, -43.6695328, 41.6428642
4: -12.5255804, 27.4433079, -14.0812511, 31.0096245, -43.5352020, 41.5245514

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.3216124, 27.1479626, -36.0727844, 40.5069618
1: -10.5382071, 37.6687279, -8.5541677, 30.6612759, -41.1994820, 46.2228966
2: -11.0602732, 37.7767830, -9.0716991, 30.8593426, -41.9196091, 46.8484802
3: -16.4940701, 38.9501419, -13.4317360, 31.5026264, -47.9966965, 52.3818741
4: -17.3545589, 37.3720055, -13.9857349, 30.8085289, -48.1630859, 51.3577385

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -7.9041195, 29.0763073, -35.5826111, 32.1717606
1: -7.5842962, 27.4015598, -9.2882051, 32.8371925, -40.4214783, 36.6897659
2: -8.0552320, 27.5818462, -9.7911968, 33.1104507, -41.1656837, 37.3730431
3: -11.9514847, 28.1189060, -14.5323257, 33.7886276, -45.7401123, 42.6512299
4: -12.5255804, 27.4433079, -15.0085173, 33.1586647, -45.6842461, 42.4518204

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -7.8569665, 28.9019203, -37.8267326, 41.0423050
1: -10.5382071, 37.6687279, -9.2307730, 32.6364670, -43.1746750, 46.8995018
2: -11.0602732, 37.7767830, -9.7337694, 32.9106445, -43.9709167, 47.5105515
3: -16.4940701, 38.9501419, -14.4427500, 33.5776863, -50.0717545, 53.3928909
4: -17.3545589, 37.3720055, -14.9150229, 32.9619713, -50.3165283, 52.2870216

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.1871862, 26.6099091, -32.7415123, 30.3283634
1: -7.1131840, 26.1596413, -8.3826809, 30.0424767, -37.1556625, 34.5423203
2: -7.5950131, 26.2545033, -8.9014492, 30.2400932, -37.8351059, 35.1559525
3: -11.2687273, 26.8291836, -13.1529007, 30.8414726, -42.1101990, 39.9820786
4: -11.9468098, 25.9775772, -13.6967449, 30.2116623, -42.1584702, 39.6743240

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.1430955, 26.4489613, -34.9817734, 39.1630821
1: -10.0470028, 36.3811836, -8.3301830, 29.8583546, -39.9053574, 44.7113647
2: -10.5810204, 36.3961143, -8.8481064, 30.0566387, -40.6376572, 45.2442207
3: -15.7773991, 37.6067963, -13.0709028, 30.6479645, -46.4253616, 50.6777000
4: -16.7466125, 35.8476906, -13.6098948, 30.0314522, -46.7780647, 49.4575844

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.7295799, 28.3989639, -34.5305672, 30.8707561
1: -7.1131840, 26.1596413, -9.0717134, 32.0508652, -39.1640472, 35.2313538
2: -7.5950131, 26.2545033, -9.5743036, 32.3409538, -39.9359665, 35.8288040
3: -11.2687273, 26.8291836, -14.1834040, 32.9562950, -44.2250214, 41.0125771
4: -11.9468098, 25.9775772, -14.6434193, 32.4114075, -44.3582153, 40.6209946

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.6920562, 28.2595654, -36.7923813, 39.7120399
1: -10.0470028, 36.3811836, -9.0262203, 31.8900490, -41.9370499, 45.4074020
2: -10.5810204, 36.3961143, -9.5286884, 32.1826820, -42.7637024, 45.9248009
3: -15.7773991, 37.6067963, -14.1123037, 32.7868042, -48.5642014, 51.7191010
4: -16.7466125, 35.8476906, -14.5677023, 32.2570839, -49.0036964, 50.4153938

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.1871862, 26.6099091, -33.2752991, 32.0736694
1: -7.7828097, 28.1194229, -8.3826809, 30.0424767, -37.8252754, 36.5021057
2: -8.2539454, 28.2906685, -8.9014492, 30.2400932, -38.4940376, 37.1921158
3: -12.2706099, 28.8829918, -13.1529007, 30.8414726, -43.1120834, 42.0358925
4: -12.8631258, 28.1254616, -13.6967449, 30.2116623, -43.0747833, 41.8222046

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.1430955, 26.4489613, -35.5605469, 41.0556374
1: -10.7727814, 38.5158424, -8.3301830, 29.8583546, -40.6311340, 46.8460236
2: -11.2928276, 38.6079216, -8.8481064, 30.0566387, -41.3494644, 47.4560280
3: -16.8676147, 39.8401985, -13.0709028, 30.6479645, -47.5155792, 52.9110985
4: -17.7444897, 38.1677361, -13.6098948, 30.0314522, -47.7759399, 51.7776260

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.7295799, 28.3989639, -35.0643578, 32.6160660
1: -7.7828097, 28.1194229, -9.0717134, 32.0508652, -39.8336678, 37.1911354
2: -8.2539454, 28.2906685, -9.5743036, 32.3409538, -40.5948982, 37.8649712
3: -12.2706099, 28.8829918, -14.1834040, 32.9562950, -45.2269058, 43.0663910
4: -12.8631258, 28.1254616, -14.6434193, 32.4114075, -45.2745285, 42.7688828

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.6920562, 28.2595654, -37.3711548, 41.6045952
1: -10.7727814, 38.5158424, -9.0262203, 31.8900490, -42.6628304, 47.5420570
2: -11.2928276, 38.6079216, -9.5286884, 32.1826820, -43.4755096, 48.1366119
3: -16.8676147, 39.8401985, -14.1123037, 32.7868042, -49.6544189, 53.9524994
4: -17.7444897, 38.1677361, -14.5677023, 32.2570839, -50.0015717, 52.7354393

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.3701210, 27.3269844, -33.4585876, 30.5112972
1: -7.1131840, 26.1596413, -8.6132603, 30.8662567, -37.9794388, 34.7728996
2: -7.5950131, 26.2545033, -9.1305141, 31.0644073, -38.6594200, 35.3850136
3: -11.2687273, 26.8291836, -13.5239582, 31.7180500, -42.9867783, 40.3531380
4: -11.9468098, 25.9775772, -14.0812511, 31.0096245, -42.9564323, 40.0588188

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.3216124, 27.1479626, -35.6807785, 39.3415985
1: -10.0470028, 36.3811836, -8.5541677, 30.6612759, -40.7082787, 44.9353523
2: -10.5810204, 36.3961143, -9.0716991, 30.8593426, -41.4403572, 45.4678116
3: -15.7773991, 37.6067963, -13.4317360, 31.5026264, -47.2800255, 51.0385323
4: -16.7466125, 35.8476906, -13.9857349, 30.8085289, -47.5551414, 49.8334274

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -7.9041195, 29.0763073, -35.2079163, 31.0452976
1: -7.1131840, 26.1596413, -9.2882051, 32.8371925, -39.9503708, 35.4478455
2: -7.5950131, 26.2545033, -9.7911968, 33.1104507, -40.7054634, 36.0457001
3: -11.2687273, 26.8291836, -14.5323257, 33.7886276, -45.0573540, 41.3615036
4: -11.9468098, 25.9775772, -15.0085173, 33.1586647, -45.1054764, 40.9860878

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -7.8569665, 28.9019203, -37.4347267, 39.8769455
1: -10.0470028, 36.3811836, -9.2307730, 32.6364670, -42.6834717, 45.6119576
2: -10.5810204, 36.3961143, -9.7337694, 32.9106445, -43.4916649, 46.1298828
3: -15.7773991, 37.6067963, -14.4427500, 33.5776863, -49.3550873, 52.0495453
4: -16.7466125, 35.8476906, -14.9150229, 32.9619713, -49.7085800, 50.7627106

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.3701210, 27.3269844, -33.9923782, 32.2566032
1: -7.7828097, 28.1194229, -8.6132603, 30.8662567, -38.6490517, 36.7326813
2: -8.2539454, 28.2906685, -9.1305141, 31.0644073, -39.3183479, 37.4211807
3: -12.2706099, 28.8829918, -13.5239582, 31.7180500, -43.9886589, 42.4069519
4: -12.8631258, 28.1254616, -14.0812511, 31.0096245, -43.8727493, 42.2067108

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.3216124, 27.1479626, -36.2595520, 41.2341537
1: -10.7727814, 38.5158424, -8.5541677, 30.6612759, -41.4340591, 47.0700111
2: -11.2928276, 38.6079216, -9.0716991, 30.8593426, -42.1521683, 47.6796188
3: -16.8676147, 39.8401985, -13.4317360, 31.5026264, -48.3702393, 53.2719345
4: -17.7444897, 38.1677361, -13.9857349, 30.8085289, -48.5530167, 52.1534729

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -7.9041195, 29.0763073, -35.7416992, 32.7906036
1: -7.7828097, 28.1194229, -9.2882051, 32.8371925, -40.6199875, 37.4076271
2: -8.2539454, 28.2906685, -9.7911968, 33.1104507, -41.3643951, 38.0818634
3: -12.2706099, 28.8829918, -14.5323257, 33.7886276, -46.0592384, 43.4153175
4: -12.8631258, 28.1254616, -15.0085173, 33.1586647, -46.0217896, 43.1339760

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.8569665, 28.9019203, -38.0135078, 41.7695007
1: -10.7727814, 38.5158424, -9.2307730, 32.6364670, -43.4092484, 47.7466164
2: -11.2928276, 38.6079216, -9.7337694, 32.9106445, -44.2034721, 48.3416901
3: -16.8676147, 39.8401985, -14.4427500, 33.5776863, -50.4453011, 54.2829475
4: -17.7444897, 38.1677361, -14.9150229, 32.9619713, -50.7064590, 53.0827560

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.33 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5702890, upper bound: 43.5766722
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5679510, upper bound: 43.5679510
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5691259, upper bound: 43.5588638
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5749305, upper bound: 43.5568833
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5749305, upper bound: 43.5568833
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5669360, upper bound: 43.5546301
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5641460, upper bound: 43.5441267
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5604451, upper bound: 43.5250272
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5532520, upper bound: 43.5230864
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5535317, upper bound: 43.5437457
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5441267, upper bound: 43.5641460
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5437457, upper bound: 43.5535317
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181094, upper bound: 43.5486156
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5181075, upper bound: 43.5463511
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -6.7334614, 25.0752354, -31.8086967, 31.8086967
1: -7.8301010, 28.3211842, -7.8301010, 28.3211842, -36.1512833, 36.1512833
2: -8.3398962, 28.4661274, -8.3398962, 28.4661274, -36.8060226, 36.8060226
3: -12.3204012, 29.0548534, -12.3204012, 29.0548534, -41.3752556, 41.3752556
4: -12.9191322, 28.3510303, -12.9191322, 28.3510303, -41.2701645, 41.2701645

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5942140, upper bound: 43.5814634
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5819614, upper bound: 43.5792965
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.3271179, 27.2405643, -33.9740257, 32.4023514
1: -7.8301010, 28.3211842, -8.5321484, 30.8015041, -38.6316071, 36.8533325
2: -8.3398962, 28.4661274, -9.0758047, 30.9408817, -39.2807770, 37.5419312
3: -12.3204012, 29.0548534, -13.4046860, 31.6406326, -43.9610329, 42.4595413
4: -12.9191322, 28.3510303, -14.0712585, 30.7746468, -43.6937752, 42.4222794

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5942140, upper bound: 43.5814634
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5819614, upper bound: 43.5792965
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -5.9322810, 22.3770065, -29.7041245, 33.1728439
1: -8.5321484, 30.8015041, -6.8647075, 25.2962532, -33.8283997, 37.6662102
2: -9.0758047, 30.9408817, -7.3448024, 25.3729000, -34.4487000, 38.2856827
3: -13.4046860, 31.6406326, -10.8725176, 25.9015713, -39.3062592, 42.5131454
4: -14.0712585, 30.7746468, -11.5375271, 25.1196098, -39.1908607, 42.3121719

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553958, upper bound: 43.5270182
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5664776, upper bound: 43.5700535
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 4

Time for candidate selection: 5.48 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5619816, upper bound: 43.5485261
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5415000, upper bound: 43.5465475
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2836843, 27.0819054, -8.3238716, 31.2092571, -38.4929390, 35.4057770
1: -8.4799194, 30.6195946, -9.7799387, 35.4459076, -43.9258270, 40.3995285
2: -9.0231104, 30.7586956, -10.3173752, 35.4635620, -44.4866714, 41.0760727
3: -13.3230801, 31.4491863, -15.3544025, 36.6176567, -49.9407349, 46.8035812
4: -13.9853888, 30.5967426, -16.3144875, 34.9340706, -48.9194565, 46.9112167

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5679510, upper bound: 43.5679510
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553257, upper bound: 43.5266298
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5227592, upper bound: 43.5227592
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.2693892, 26.8420372, -33.5755005, 32.3446236
1: -7.8301010, 28.3211842, -8.5089540, 30.3026695, -38.1327705, 36.8301392
2: -8.3398962, 28.4661274, -9.0044117, 30.5334988, -38.8733940, 37.4705391
3: -12.3204012, 29.0548534, -13.3364210, 31.1392040, -43.4596024, 42.3912735
4: -12.9191322, 28.3510303, -13.8503227, 30.5232334, -43.4423637, 42.2013550

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5918492, upper bound: 43.5637368
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5807983, upper bound: 43.5614882
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.8313341, 28.8915958, -35.6250572, 32.9065704
1: -7.8301010, 28.3211842, -9.1737223, 32.6585579, -40.4886589, 37.4949036
2: -8.3398962, 28.4661274, -9.7022839, 32.8745689, -41.2144661, 38.1684113
3: -12.3204012, 29.0548534, -14.3621759, 33.5927696, -45.9131699, 43.4170265
4: -12.9191322, 28.3510303, -14.9443817, 32.8122597, -45.7313843, 43.2954102

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5918492, upper bound: 43.5637368
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5807983, upper bound: 43.5614882
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.4716120, 24.1467953, -31.4739113, 33.7121735
1: -8.5321484, 30.8015041, -7.5406446, 27.2661324, -35.7982788, 38.3421478
2: -9.0758047, 30.9408817, -8.0123167, 27.4411392, -36.5169373, 38.9531975
3: -13.4046860, 31.6406326, -11.8849792, 27.9768200, -41.3815041, 43.5256081
4: -14.0712585, 30.7746468, -12.4636946, 27.2980309, -41.3692741, 43.2383385

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2836843, 27.0819054, -8.8989973, 33.0930710, -40.3767548, 35.9809036
1: -8.4799194, 30.6195946, -10.5058804, 37.5652847, -46.0452042, 41.1254730
2: -9.0231104, 30.7586956, -11.0282240, 37.6693153, -46.6924248, 41.7869186
3: -13.3230801, 31.4491863, -16.4446697, 38.8422585, -52.1653328, 47.8938484
4: -13.9853888, 30.5967426, -17.3083668, 37.2609673, -51.2463493, 47.9051056

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -5.9647384, 22.4902878, -28.9965916, 30.2323761
1: -7.5842962, 27.4015598, -6.9048386, 25.4210529, -33.0053444, 34.3063965
2: -8.0552320, 27.5818462, -7.3848820, 25.5047226, -33.5599556, 34.9667244
3: -11.9514847, 28.1189060, -10.9337454, 26.0330429, -37.9845276, 39.0526505
4: -12.5255804, 27.4433079, -11.5946884, 25.2559605, -37.7815361, 39.0379829

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589390, upper bound: 43.5691566
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588638, upper bound: 43.5691259
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.3492012, 31.2999210, -37.8062248, 32.6168404
1: -7.5842962, 27.4015598, -9.8115788, 35.5474586, -43.1317520, 37.2131348
2: -8.0552320, 27.5818462, -10.3486872, 35.5690842, -43.6243172, 37.9305344
3: -11.9514847, 28.1189060, -15.4028683, 36.7234039, -48.6748848, 43.5217743
4: -12.5255804, 27.4433079, -16.3597260, 35.0432816, -47.5688629, 43.8030281

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589390, upper bound: 43.5691566
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588638, upper bound: 43.5691259
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -5.9647384, 22.4902878, -31.4151058, 39.1500854
1: -10.5382071, 37.6687279, -6.9048386, 25.4210529, -35.9592590, 44.5735664
2: -11.0602732, 37.7767830, -7.3848820, 25.5047226, -36.5649910, 45.1616669
3: -16.4940701, 38.9501419, -10.9337454, 26.0330429, -42.5271149, 49.8838844
4: -17.3545589, 37.3720055, -11.5946884, 25.2559605, -42.6105156, 48.9666862

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5521994, upper bound: 43.5265669
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5228568
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.2806950, 31.0336685, -39.9584885, 41.4660378
1: -10.5382071, 37.6687279, -9.6846685, 35.2271957, -45.7653999, 47.3533974
2: -11.0602732, 37.7767830, -10.2633362, 35.2591934, -46.3194656, 48.0401192
3: -16.4940701, 38.9501419, -15.2045250, 36.3579483, -52.8520164, 54.1546669
4: -17.3545589, 37.3720055, -16.2140293, 34.7246552, -52.0792160, 53.5860329

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5521994, upper bound: 43.5265669
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5228568
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.5063033, 24.2676392, -30.7739429, 30.7739429
1: -7.5842962, 27.4015598, -7.5842962, 27.4015598, -34.9858475, 34.9858475
2: -8.0552320, 27.5818462, -8.0552320, 27.5818462, -35.6370773, 35.6370773
3: -11.9514847, 28.1189060, -11.9514847, 28.1189060, -40.0703888, 40.0703888
4: -12.5255804, 27.4433079, -12.5255804, 27.4433079, -39.9688873, 39.9688873

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585932, upper bound: 43.5579472
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585180, upper bound: 43.5579165
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.9248209, 33.1853523, -39.6916504, 33.1924591
1: -7.5842962, 27.4015598, -10.5382071, 37.6687279, -45.2530251, 37.9397621
2: -8.0552320, 27.5818462, -11.0602732, 37.7767830, -45.8320160, 38.6421165
3: -11.9514847, 28.1189060, -16.4940701, 38.9501419, -50.9016266, 44.6129761
4: -12.5255804, 27.4433079, -17.3545589, 37.3720055, -49.8975830, 44.7978668

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585932, upper bound: 43.5579472
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585180, upper bound: 43.5579165
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.5063033, 24.2676392, -33.1924591, 39.6916504
1: -10.5382071, 37.6687279, -7.5842962, 27.4015598, -37.9397621, 45.2530251
2: -11.0602732, 37.7767830, -8.0552320, 27.5818462, -38.6421165, 45.8320160
3: -16.4940701, 38.9501419, -11.9514847, 28.1189060, -44.6129761, 50.9016266
4: -17.3545589, 37.3720055, -12.5255804, 27.4433079, -44.7978630, 49.8975830

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5521994, upper bound: 43.5260603
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5223382
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.9248209, 33.1853523, -42.1101685, 42.1101685
1: -10.5382071, 37.6687279, -10.5382071, 37.6687279, -48.2069359, 48.2069359
2: -11.0602732, 37.7767830, -11.0602732, 37.7767830, -48.8370552, 48.8370552
3: -16.4940701, 38.9501419, -16.4940701, 38.9501419, -55.4442139, 55.4442139
4: -17.3545589, 37.3720055, -17.3545589, 37.3720055, -54.7265625, 54.7265625

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5521994, upper bound: 43.5260603
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5223382
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -6.9100652, 25.7693195, -32.5027809, 31.9853001
1: -7.8301010, 28.3211842, -8.0518236, 29.1186275, -36.9487305, 36.3730049
2: -8.3398962, 28.4661274, -8.5616179, 29.2663651, -37.6062622, 37.0277443
3: -12.3204012, 29.0548534, -12.6775093, 29.9030113, -42.2234116, 41.7323608
4: -12.9191322, 28.3510303, -13.2914982, 29.1219196, -42.0410385, 41.6425247

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5860737, upper bound: 43.5588651
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5767913, upper bound: 43.5571490
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.5059209, 27.9394646, -34.6729279, 32.5811577
1: -7.8301010, 28.3211842, -8.7573853, 31.6072998, -39.4374008, 37.0785637
2: -8.3398962, 28.4661274, -9.2995558, 31.7472954, -40.0871925, 37.7656822
3: -12.3204012, 29.0548534, -13.7668066, 32.4964180, -44.8168182, 42.8216591
4: -12.9191322, 28.3510303, -14.4473610, 31.5514050, -44.4705353, 42.7983932

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5860737, upper bound: 43.5588651
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5767913, upper bound: 43.5571490
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.1033306, 23.0420818, -30.3691978, 33.3438911
1: -8.5321484, 30.8015041, -7.0784302, 26.0490246, -34.5811729, 37.8799362
2: -9.0758047, 30.9408817, -7.5601616, 26.1392612, -35.2150650, 38.5010452
3: -13.4046860, 31.6406326, -11.2156792, 26.7142754, -40.1189613, 42.8563080
4: -14.0712585, 30.7746468, -11.8969269, 25.8581009, -39.9293518, 42.6715660

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5641460, upper bound: 43.5441267
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5641460, upper bound: 43.5441267
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2836843, 27.0819054, -8.5076342, 31.9295826, -39.2132683, 35.5895386
1: -8.4799194, 30.6195946, -10.0155573, 36.2799606, -44.7598801, 40.6351509
2: -9.0231104, 30.7586956, -10.5499544, 36.2909279, -45.3140373, 41.3086510
3: -13.3230801, 31.4491863, -15.7292099, 37.5016785, -50.8247604, 47.1783943
4: -13.9853888, 30.5967426, -16.7020378, 35.7388153, -49.7242012, 47.2987747

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5641460, upper bound: 43.5441267
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5641460, upper bound: 43.5441267
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -7.4370813, 27.4980087, -34.2314682, 32.5123177
1: -7.8301010, 28.3211842, -8.7175827, 31.0629864, -38.8930893, 37.0387650
2: -8.3398962, 28.4661274, -9.2134495, 31.2856255, -39.6255226, 37.6795769
3: -12.3204012, 29.0548534, -13.6725578, 31.9439163, -44.2643166, 42.7274055
4: -12.9191322, 28.3510303, -14.2024050, 31.2481651, -44.1672935, 42.5534363

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5663383, upper bound: 43.5264556
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5620862, upper bound: 43.5252338
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7334614, 25.0752354, -8.0359106, 29.6763287, -36.4097900, 33.1111450
1: -7.8301010, 28.3211842, -9.4270086, 33.5660286, -41.3961296, 37.7481918
2: -8.3398962, 28.4661274, -9.9561176, 33.7784920, -42.1183891, 38.4222450
3: -12.3204012, 29.0548534, -14.7677183, 34.5513916, -46.8717918, 43.8225708
4: -12.9191322, 28.3510303, -15.3659382, 33.6877899, -46.6069221, 43.7169685

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5663383, upper bound: 43.5264556
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5620862, upper bound: 43.5252338
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3271179, 27.2405643, -6.6372471, 24.7875099, -32.1146240, 33.8778076
1: -8.5321484, 30.8015041, -7.7472224, 28.0086174, -36.5407639, 38.5487251
2: -9.0758047, 30.9408817, -8.2193661, 28.1753578, -37.2511520, 39.1602440
3: -13.4046860, 31.6406326, -12.2163754, 28.7673283, -42.1720123, 43.8570061
4: -14.0712585, 30.7746468, -12.8135014, 28.0057850, -42.0770264, 43.5881424

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2836843, 27.0819054, -9.0855732, 33.8189926, -41.1026764, 36.1674805
1: -8.4799194, 30.6195946, -10.7402515, 38.4109764, -46.8908958, 41.3598480
2: -9.0231104, 30.7586956, -11.2606144, 38.4991837, -47.5222931, 42.0193100
3: -13.3230801, 31.4491863, -16.8177757, 39.7311020, -53.0541801, 48.2669601
4: -13.9853888, 30.5967426, -17.6979370, 38.0556946, -52.0410805, 48.2946777

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.1316075, 23.1411781, -29.6474819, 30.3992424
1: -7.5842962, 27.4015598, -7.1131840, 26.1596413, -33.7439346, 34.5147400
2: -8.0552320, 27.5818462, -7.5950131, 26.2545033, -34.3097343, 35.1768570
3: -11.9514847, 28.1189060, -11.2687273, 26.8291836, -38.7806625, 39.3876343
4: -12.5255804, 27.4433079, -11.9468098, 25.9775772, -38.5031548, 39.3901176

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555188, upper bound: 43.5452950
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5554495, upper bound: 43.5452343
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.5328178, 32.0199852, -38.5262871, 32.8004570
1: -7.5842962, 27.4015598, -10.0470028, 36.3811836, -43.9654770, 37.4485626
2: -8.0552320, 27.5818462, -10.5810204, 36.3961143, -44.4513474, 38.1628647
3: -11.9514847, 28.1189060, -15.7773991, 37.6067963, -49.5582809, 43.8963051
4: -12.5255804, 27.4433079, -16.7466125, 35.8476906, -48.3732719, 44.1899185

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555188, upper bound: 43.5452950
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5554495, upper bound: 43.5452343
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.1316075, 23.1411781, -32.0659981, 39.3169518
1: -10.5382071, 37.6687279, -7.1131840, 26.1596413, -36.6978455, 44.7819138
2: -11.0602732, 37.7767830, -7.5950131, 26.2545033, -37.3147774, 45.3717957
3: -16.4940701, 38.9501419, -11.2687273, 26.8291836, -43.3232498, 50.2188683
4: -17.3545589, 37.3720055, -11.9468098, 25.9775772, -43.3321342, 49.3188133

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515296, upper bound: 43.5263496
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218139, upper bound: 43.5226517
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.5328178, 32.0199852, -40.9448051, 41.7181664
1: -10.5382071, 37.6687279, -10.0470028, 36.3811836, -46.9193916, 47.7157288
2: -11.0602732, 37.7767830, -10.5810204, 36.3961143, -47.4563866, 48.3578033
3: -16.4940701, 38.9501419, -15.7773991, 37.6067963, -54.1008682, 54.7275391
4: -17.3545589, 37.3720055, -16.7466125, 35.8476906, -53.2022476, 54.1186142

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515296, upper bound: 43.5263496
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218139, upper bound: 43.5226517
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.6653934, 24.8864861, -31.3927898, 30.9330311
1: -7.5842962, 27.4015598, -7.7828097, 28.1194229, -35.7037201, 35.1843605
2: -8.0552320, 27.5818462, -8.2539454, 28.2906685, -36.3459015, 35.8357849
3: -11.9514847, 28.1189060, -12.2706099, 28.8829918, -40.8344765, 40.3895149
4: -12.5255804, 27.4433079, -12.8631258, 28.1254616, -40.6510429, 40.3064346

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5485978, upper bound: 43.5195879
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5484986, upper bound: 43.5194712
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -9.1115875, 33.9125404, -40.4188423, 33.3792267
1: -7.5842962, 27.4015598, -10.7727814, 38.5158424, -46.1001358, 38.1743393
2: -8.0552320, 27.5818462, -11.2928276, 38.6079216, -46.6631546, 38.8746719
3: -11.9514847, 28.1189060, -16.8676147, 39.8401985, -51.7916832, 44.9865189
4: -12.5255804, 27.4433079, -17.7444897, 38.1677361, -50.6933174, 45.1877975

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5485978, upper bound: 43.5195879
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5484986, upper bound: 43.5194712
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.6653934, 24.8864861, -33.8113060, 39.8507423
1: -10.5382071, 37.6687279, -7.7828097, 28.1194229, -38.6576309, 45.4515343
2: -11.0602732, 37.7767830, -8.2539454, 28.2906685, -39.3509407, 46.0307274
3: -16.4940701, 38.9501419, -12.2706099, 28.8829918, -45.3770599, 51.2207527
4: -17.3545589, 37.3720055, -12.8631258, 28.1254616, -45.4800186, 50.2351265

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210155, upper bound: 43.5147703
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -9.1115875, 33.9125404, -42.8373604, 42.2969398
1: -10.5382071, 37.6687279, -10.7727814, 38.5158424, -49.0540466, 48.4415092
2: -11.0602732, 37.7767830, -11.2928276, 38.6079216, -49.6681938, 49.0696106
3: -16.4940701, 38.9501419, -16.8676147, 39.8401985, -56.3342667, 55.8177567
4: -17.3545589, 37.3720055, -17.7444897, 38.1677361, -55.5222931, 55.1164932

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210155, upper bound: 43.5147703
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -5.9647384, 22.4902878, -28.6218910, 29.1059151
1: -7.1131840, 26.1596413, -6.9048386, 25.4210529, -32.5342369, 33.0644798
2: -7.5950131, 26.2545033, -7.3848820, 25.5047226, -33.0997314, 33.6393814
3: -11.2687273, 26.8291836, -10.9337454, 26.0330429, -37.3017693, 37.7629204
4: -11.9468098, 25.9775772, -11.5946884, 25.2559605, -37.2027702, 37.5722542

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5678573, upper bound: 43.5700808
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546301, upper bound: 43.5669360
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -8.3492012, 31.2999210, -37.4315300, 31.4903793
1: -7.1131840, 26.1596413, -9.8115788, 35.5474586, -42.6606445, 35.9712219
2: -7.5950131, 26.2545033, -10.3486872, 35.5690842, -43.1640968, 36.6031914
3: -11.2687273, 26.8291836, -15.4028683, 36.7234039, -47.9921303, 42.2320442
4: -11.9468098, 25.9775772, -16.3597260, 35.0432816, -46.9900894, 42.3372993

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5678573, upper bound: 43.5700808
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546301, upper bound: 43.5669360
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -5.9647384, 22.4902878, -31.0231018, 37.9847221
1: -10.0470028, 36.3811836, -6.9048386, 25.4210529, -35.4680557, 43.2860222
2: -10.5810204, 36.3961143, -7.3848820, 25.5047226, -36.0857315, 43.7809944
3: -15.7773991, 37.6067963, -10.9337454, 26.0330429, -41.8104401, 48.5405426
4: -16.7466125, 35.8476906, -11.5946884, 25.2559605, -42.0025711, 47.4423752

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5235777
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5223325
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -8.2806950, 31.0336685, -39.5664825, 40.3006783
1: -10.0470028, 36.3811836, -9.6846685, 35.2271957, -45.2742004, 46.0658531
2: -10.5810204, 36.3961143, -10.2633362, 35.2591934, -45.8402138, 46.6594505
3: -15.7773991, 37.6067963, -15.2045250, 36.3579483, -52.1353455, 52.8113213
4: -16.7466125, 35.8476906, -16.2140293, 34.7246552, -51.4712677, 52.0617218

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5235777
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5223325
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -6.5063033, 24.2676392, -30.3992424, 29.6474819
1: -7.1131840, 26.1596413, -7.5842962, 27.4015598, -34.5147400, 33.7439308
2: -7.5950131, 26.2545033, -8.0552320, 27.5818462, -35.1768608, 34.3097343
3: -11.2687273, 26.8291836, -11.9514847, 28.1189060, -39.3876343, 38.7806625
4: -11.9468098, 25.9775772, -12.5255804, 27.4433079, -39.3901138, 38.5031509

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5589374
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542407, upper bound: 43.5560581
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1316075, 23.1411781, -8.9248209, 33.1853523, -39.3169556, 32.0659981
1: -7.1131840, 26.1596413, -10.5382071, 37.6687279, -44.7819138, 36.6978455
2: -7.5950131, 26.2545033, -11.0602732, 37.7767830, -45.3717957, 37.3147774
3: -11.2687273, 26.8291836, -16.4940701, 38.9501419, -50.2188683, 43.3232498
4: -11.9468098, 25.9775772, -17.3545589, 37.3720055, -49.3188133, 43.3321304

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5589374
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542407, upper bound: 43.5560581
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -6.5063033, 24.2676392, -32.8004570, 38.5262871
1: -10.0470028, 36.3811836, -7.5842962, 27.4015598, -37.4485626, 43.9654808
2: -10.5810204, 36.3961143, -8.0552320, 27.5818462, -38.1628647, 44.4513474
3: -15.7773991, 37.6067963, -11.9514847, 28.1189060, -43.8963051, 49.5582809
4: -16.7466125, 35.8476906, -12.5255804, 27.4433079, -44.1899185, 48.3732719

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5400465, upper bound: 43.5230592
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5226517, upper bound: 43.5218139
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.5328178, 32.0199852, -8.9248209, 33.1853523, -41.7181664, 40.9448051
1: -10.0470028, 36.3811836, -10.5382071, 37.6687279, -47.7157288, 46.9193916
2: -10.5810204, 36.3961143, -11.0602732, 37.7767830, -48.3578033, 47.4563866
3: -15.7773991, 37.6067963, -16.4940701, 38.9501419, -54.7275391, 54.1008682
4: -16.7466125, 35.8476906, -17.3545589, 37.3720055, -54.1186142, 53.2022476

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5400465, upper bound: 43.5230592
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5218139
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -5.9647384, 22.4902878, -29.1556797, 30.8512211
1: -7.7828097, 28.1194229, -6.9048386, 25.4210529, -33.2038612, 35.0242615
2: -8.2539454, 28.2906685, -7.3848820, 25.5047226, -33.7586594, 35.6755524
3: -12.2706099, 28.8829918, -10.9337454, 26.0330429, -38.3036537, 39.8167343
4: -12.8631258, 28.1254616, -11.5946884, 25.2559605, -38.1190834, 39.7201462

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255612, upper bound: 43.5576939
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5230864, upper bound: 43.5532520
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.6653934, 24.8864861, -8.3492012, 31.2999210, -37.9653130, 33.2356834
1: -7.7828097, 28.1194229, -9.8115788, 35.5474586, -43.3302612, 37.9309998
2: -8.2539454, 28.2906685, -10.3486872, 35.5690842, -43.8230286, 38.6393547
3: -12.2706099, 28.8829918, -15.4028683, 36.7234039, -48.9940109, 44.2858582
4: -12.8631258, 28.1254616, -16.3597260, 35.0432816, -47.9064064, 44.4851875

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255612, upper bound: 43.5576939
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5230864, upper bound: 43.5532520
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -5.9647384, 22.4902878, -31.6018753, 39.8772774
1: -10.7727814, 38.5158424, -6.9048386, 25.4210529, -36.1938324, 45.4206810
2: -11.2928276, 38.6079216, -7.3848820, 25.5047226, -36.7975464, 45.9928055
3: -16.8676147, 39.8401985, -10.9337454, 26.0330429, -42.9006577, 50.7739410
4: -17.7444897, 38.1677361, -11.5946884, 25.2559605, -43.0004463, 49.7624207

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 46

Time for candidate selection: 3.81 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165911, upper bound: 43.5454005
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5153342, upper bound: 43.5443180
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -8.2806950, 31.0336685, -40.1452560, 42.1932297
1: -10.7727814, 38.5158424, -9.6846685, 35.2271957, -45.9999771, 48.2005119
2: -11.2928276, 38.6079216, -10.2633362, 35.2591934, -46.5520210, 48.8712578
3: -16.8676147, 39.8401985, -15.2045250, 36.3579483, -53.2255592, 55.0447235
4: -17.7444897, 38.1677361, -16.2140293, 34.7246552, -52.4691467, 54.3817673

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 46

Time for candidate selection: 4.08 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5902231, upper bound: 43.5770557
time: 0.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760295, upper bound: 43.5760295
time: 0.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 3, lower bound: -43.5902231, upper bound: 43.5770557
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 3, lower bound: -43.5760295, upper bound: 43.5760295

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.6070757, 28.1151905, -35.5684166, 35.1776581
1: -8.7061062, 31.1410255, -8.8946314, 31.7592926, -40.4653969, 40.0356560
2: -9.2308626, 31.3365688, -9.4227009, 31.9674187, -41.1982803, 40.7592697
3: -13.6493845, 31.9803791, -13.9369183, 32.6282806, -46.2776642, 45.9172974
4: -14.1980610, 31.2951145, -14.4853649, 31.9250107, -46.1230698, 45.7804756

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760295, upper bound: 43.5760295
time: 0.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760295, upper bound: 43.5760295
time: 0.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.6380959, 28.2877064, -7.5247240, 27.8562355, -35.4943314, 35.8124313
1: -8.9388247, 31.9677963, -8.7925901, 31.4697495, -40.4085655, 40.7603874
2: -9.4615984, 32.1662331, -9.3212376, 31.6661911, -41.1277885, 41.4874687
3: -14.0235004, 32.8585358, -13.7896824, 32.3263931, -46.3498917, 46.6482124
4: -14.5844440, 32.0941963, -14.3478031, 31.6118011, -46.1962433, 46.4419937

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5284495, upper bound: 43.5587476
time: 0.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255237, upper bound: 43.5255237
time: 0.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5760295, upper bound: 43.5760295
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5760295, upper bound: 43.5760295
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5284495, upper bound: 43.5587476
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5255237, upper bound: 43.5255237

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.4532280, 27.5705853, -35.0238037, 35.0238037
1: -8.7061062, 31.1410255, -8.7061062, 31.1410255, -39.8471298, 39.8471298
2: -9.2308626, 31.3365688, -9.2308626, 31.3365688, -40.5674324, 40.5674324
3: -13.6493845, 31.9803791, -13.6493845, 31.9803791, -45.6297646, 45.6297646
4: -14.1980610, 31.2951145, -14.1980610, 31.2951145, -45.4931755, 45.4931755

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577762, upper bound: 43.5279890
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457657, upper bound: 43.5267904
time: 0.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.4532280, 27.5705853, -7.6380959, 28.2877064, -35.7409325, 35.2086716
1: -8.7061062, 31.1410255, -8.9388247, 31.9677963, -40.6739006, 40.0798416
2: -9.2308626, 31.3365688, -9.4615984, 32.1662331, -41.3970947, 40.7981682
3: -13.6493845, 31.9803791, -14.0235004, 32.8585358, -46.5079193, 46.0038795
4: -14.1980610, 31.2951145, -14.5844440, 32.0941963, -46.2922554, 45.8795586

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577762, upper bound: 43.5279890
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457657, upper bound: 43.5267904
time: 0.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.5718665, 28.0498104, -7.2623110, 26.9129429, -34.4848099, 35.3121109
1: -8.8584318, 31.6948280, -8.4736862, 30.3897781, -39.2482109, 40.1685028
2: -9.3797674, 31.8935356, -8.9968538, 30.5858803, -39.9656487, 40.8903885
3: -13.9000330, 32.5757828, -13.3001461, 31.2071037, -45.1071358, 45.8759308
4: -14.4596872, 31.8262424, -13.8543615, 30.5485668, -45.0082550, 45.6805954

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5209298, upper bound: 43.5376089
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.53 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.5117970, 27.8515053, -7.8049746, 28.6958961, -36.2076874, 35.6564751
1: -8.7829199, 31.4712200, -9.1622686, 32.3940964, -41.1770096, 40.6334763
2: -9.3046150, 31.6622829, -9.6685925, 32.6785431, -41.9831581, 41.3308716
3: -13.7871284, 32.3372002, -14.3298416, 33.3161049, -47.1032333, 46.6670418
4: -14.3511839, 31.5855846, -14.7992086, 32.7390366, -47.0902214, 46.3847847

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5193028, upper bound: 43.5168084
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5577762, upper bound: 43.5279890
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5457657, upper bound: 43.5267904
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5577762, upper bound: 43.5279890
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5457657, upper bound: 43.5267904
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5209298, upper bound: 43.5376089
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5193028, upper bound: 43.5168084
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.3853736, 27.3256817, -34.5128632, 33.9952812
1: -8.3826809, 30.0424767, -8.6235952, 30.8608303, -39.2435112, 38.6660728
2: -8.9014492, 30.2400932, -9.1469164, 31.0559006, -39.9573479, 39.3870087
3: -13.1529007, 30.8414726, -13.5226707, 31.6899166, -44.8428192, 44.3641434
4: -13.6967449, 30.2116623, -14.0700741, 31.0189495, -44.7156906, 44.2817307

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5688609, upper bound: 43.5597650
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.3203273, 27.1139317, -34.8435135, 35.7192917
1: -9.0717134, 32.0508652, -8.5420380, 30.6224041, -39.6941071, 40.5929031
2: -9.5743036, 32.3409538, -9.0655575, 30.8094730, -40.3837738, 41.4065094
3: -14.1834040, 32.9562950, -13.4013624, 31.4356136, -45.6190109, 46.3576546
4: -14.6434193, 32.4114075, -13.9541807, 30.7608356, -45.4042549, 46.3655891

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581231, upper bound: 43.5594560
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1871862, 26.6099091, -7.5718665, 28.0498104, -35.2369957, 34.1817741
1: -8.3826809, 30.0424767, -8.8584318, 31.6948280, -40.0775032, 38.9009056
2: -8.9014492, 30.2400932, -9.3797674, 31.8935356, -40.7949829, 39.6198616
3: -13.1529007, 30.8414726, -13.9000330, 32.5757828, -45.7286835, 44.7415047
4: -13.6967449, 30.2116623, -14.4596872, 31.8262424, -45.5229836, 44.6713486

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5394532, upper bound: 43.5224029
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.7295799, 28.3989639, -7.5117970, 27.8515053, -35.5810852, 35.9107552
1: -9.0717134, 32.0508652, -8.7829199, 31.4712200, -40.5429230, 40.8337860
2: -9.5743036, 32.3409538, -9.3046150, 31.6622829, -41.2365875, 41.6455688
3: -14.1834040, 32.9562950, -13.7871284, 32.3372002, -46.5206032, 46.7434196
4: -14.6434193, 32.4114075, -14.3511839, 31.5855846, -46.2290001, 46.7625923

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5224118
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.3281989, 23.8465290, -7.2010474, 26.7069187, -33.0351181, 31.0475769
1: -7.3499689, 26.9663219, -8.3991442, 30.1580143, -37.5079842, 35.3654633
2: -7.8391385, 27.0614948, -8.9208870, 30.3475361, -38.1866684, 35.9823837
3: -11.6318560, 27.6643848, -13.1882792, 30.9660606, -42.5979156, 40.8526573
4: -12.3158455, 26.7735996, -13.7485809, 30.3009682, -42.6168137, 40.5221748

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5209298, upper bound: 43.5376089
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5209298, upper bound: 43.5376089
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.7499533, 32.7939873, -7.1678724, 26.5654907, -35.3154449, 39.9618607
1: -10.3097925, 37.2698402, -8.3603153, 29.9914627, -40.3012543, 45.6301575
2: -10.8480654, 37.2827568, -8.8826380, 30.1904125, -41.0384750, 46.1653938
3: -16.1813354, 38.5235176, -13.1232204, 30.7888947, -46.9702301, 51.6467361
4: -17.1520309, 36.7178192, -13.6678810, 30.1618710, -47.3139000, 50.3856964

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.2679539, 23.6466656, -7.7441173, 28.4909096, -34.7588654, 31.3907814
1: -7.2751541, 26.7506466, -9.0881224, 32.1634331, -39.4385872, 35.8387680
2: -7.7640185, 26.8291969, -9.5931864, 32.4416504, -40.2056694, 36.4223747
3: -11.5195141, 27.4249935, -14.2185488, 33.0760155, -44.5955200, 41.6435318
4: -12.2073965, 26.5305767, -14.6934137, 32.4928017, -44.7001915, 41.2239914

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5193028, upper bound: 43.5168084
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5193028, upper bound: 43.5168084
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.6865768, 32.5887604, -7.7182298, 28.3753624, -37.0619354, 40.3069916
1: -10.2309980, 37.0391312, -9.0569992, 32.0229836, -42.2539825, 46.0961304
2: -10.7693300, 37.0428123, -9.5632429, 32.3130226, -43.0823517, 46.6060562
3: -16.0634308, 38.2776337, -14.1654797, 32.9252281, -48.9886551, 52.4431114
4: -17.0398197, 36.4654579, -14.6248865, 32.3823776, -49.4221954, 51.0903435

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.20 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5688609, upper bound: 43.5597650
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5581231, upper bound: 43.5594560
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5394532, upper bound: 43.5224029
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5224118
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5209298, upper bound: 43.5376089
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5209298, upper bound: 43.5376089
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5193028, upper bound: 43.5168084
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5193028, upper bound: 43.5168084
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1264782, 26.4055786, -6.1583133, 23.1884537, -30.3149319, 32.5638847
1: -8.3087387, 29.8127136, -7.1370277, 26.2245445, -34.5332756, 36.9497414
2: -8.8261719, 30.0036449, -7.6254220, 26.3030167, -35.1291885, 37.6290627
3: -13.0419569, 30.6027317, -11.2898073, 26.8580627, -39.9000130, 41.8925400
4: -13.5922346, 29.9658051, -11.9580927, 26.0414906, -39.6337242, 41.9238968

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.0997167, 26.2901516, -8.5555115, 32.0344582, -39.1341743, 34.8456573
1: -8.2793884, 29.6757069, -10.0620441, 36.3880119, -44.6674004, 39.7377510
2: -8.7958736, 29.8768692, -10.6037455, 36.4118919, -45.2077560, 40.4806137
3: -12.9914083, 30.4564915, -15.7867413, 37.5937805, -50.5851860, 46.2432175
4: -13.5233612, 29.8568954, -16.7442265, 35.8790398, -49.4024010, 46.6011200

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.6691399, 28.1956158, -6.0948410, 22.9826908, -30.6518307, 34.2904510
1: -8.9981260, 31.8220196, -7.0591693, 26.0043926, -35.0025177, 38.8811874
2: -9.4992990, 32.1057816, -7.5463943, 26.0642319, -35.5635185, 39.6521721
3: -14.0730476, 32.7179680, -11.1737213, 26.6180305, -40.6910667, 43.8916855
4: -14.5382710, 32.1667023, -11.8461428, 25.7902393, -40.3285027, 44.0128441

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.6482582, 28.0976791, -8.4854784, 31.8138542, -39.4621124, 36.5831566
1: -8.9732761, 31.7023220, -9.9755898, 36.1475639, -45.1208420, 41.6779099
2: -9.4754496, 31.9984379, -10.5169134, 36.1537857, -45.6292343, 42.5153503
3: -14.0295019, 32.5887375, -15.6582737, 37.3285637, -51.3580666, 48.2470055
4: -14.4787979, 32.0775986, -16.6212826, 35.6058006, -50.0845985, 48.6988831

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1264782, 26.4055786, -6.3281989, 23.8465290, -30.9730072, 32.7337761
1: -8.3087387, 29.8127136, -7.3499689, 26.9663219, -35.2750587, 37.1626816
2: -8.8261719, 30.0036449, -7.8391385, 27.0614948, -35.8876648, 37.8427734
3: -13.0419569, 30.6027317, -11.6318560, 27.6643848, -40.7063370, 42.2345886
4: -13.5922346, 29.9658051, -12.3158455, 26.7735996, -40.3658333, 42.2816505

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.0997167, 26.2901516, -8.7499533, 32.7939873, -39.8937035, 35.0401001
1: -8.2793884, 29.6757069, -10.3097925, 37.2698402, -45.5492249, 39.9855003
2: -8.7958736, 29.8768692, -10.8480654, 37.2827568, -46.0786285, 40.7249260
3: -12.9914083, 30.4564915, -16.1813354, 38.5235176, -51.5149231, 46.6378250
4: -13.5233612, 29.8568954, -17.1520309, 36.7178192, -50.2411804, 47.0089264

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.6691399, 28.1956158, -6.2679539, 23.6466656, -31.3158054, 34.4635658
1: -8.9981260, 31.8220196, -7.2751541, 26.7506466, -35.7487717, 39.0971756
2: -9.4992990, 32.1057816, -7.7640185, 26.8291969, -36.3284798, 39.8698006
3: -14.0730476, 32.7179680, -11.5195141, 27.4249935, -41.4980354, 44.2374763
4: -14.5382710, 32.1667023, -12.2073965, 26.5305767, -41.0688438, 44.3740921

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.6482582, 28.0976791, -8.6865768, 32.5887604, -40.2370186, 36.7842484
1: -8.9732761, 31.7023220, -10.2309980, 37.0391312, -46.0124054, 41.9333191
2: -9.4754496, 31.9984379, -10.7693300, 37.0428123, -46.5182610, 42.7677689
3: -14.0295019, 32.5887375, -16.0634308, 38.2776337, -52.3071365, 48.6521683
4: -14.4787979, 32.0775986, -17.0398197, 36.4654579, -50.9442558, 49.1174164

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.3281989, 23.8465290, -7.1264782, 26.4055786, -32.7337723, 30.9730072
1: -7.3499689, 26.9663219, -8.3087387, 29.8127136, -37.1626816, 35.2750549
2: -7.8391385, 27.0614948, -8.8261719, 30.0036449, -37.8427734, 35.8876648
3: -11.6318560, 27.6643848, -13.0419569, 30.6027317, -42.2345886, 40.7063370
4: -12.3158455, 26.7735996, -13.5922346, 29.9658051, -42.2816505, 40.3658333

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.3281989, 23.8465290, -7.3088627, 27.1210747, -33.4492722, 31.1553917
1: -7.3499689, 26.9663219, -8.5386496, 30.6346645, -37.9846344, 35.5049667
2: -7.8391385, 27.0614948, -9.0545464, 30.8270340, -38.6661720, 36.1160393
3: -11.6318560, 27.6643848, -13.4119902, 31.4771461, -43.1089973, 41.0763664
4: -12.3158455, 26.7735996, -13.9754810, 30.7618046, -43.0776482, 40.7490807

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.7499533, 32.7939873, -7.0997167, 26.2901516, -35.0401001, 39.8937035
1: -10.3097925, 37.2698402, -8.2793884, 29.6757069, -39.9855003, 45.5492249
2: -10.8480654, 37.2827568, -8.7958736, 29.8768692, -40.7249260, 46.0786285
3: -16.1813354, 38.5235176, -12.9914083, 30.4564915, -46.6378250, 51.5149231
4: -17.1520309, 36.7178192, -13.5233612, 29.8568954, -47.0089264, 50.2411804

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7499533, 32.7939873, -7.2589397, 26.9156380, -35.6655922, 40.0529251
1: -10.3097925, 37.2698402, -8.4761724, 30.3951454, -40.7049370, 45.7460136
2: -10.8480654, 37.2827568, -8.9955788, 30.5927124, -41.4407768, 46.2783318
3: -16.1813354, 38.5235176, -13.3101597, 31.2216206, -47.4029541, 51.8336792
4: -17.1520309, 36.7178192, -13.8618517, 30.5465946, -47.6986237, 50.5796661

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.2679539, 23.6466656, -7.6696191, 28.1973267, -34.4652786, 31.3162842
1: -7.2751541, 26.7506466, -8.9987154, 31.8240471, -39.0992012, 35.7493629
2: -7.7640185, 26.8291969, -9.4998446, 32.1077271, -39.8717461, 36.3290367
3: -11.5195141, 27.4249935, -14.0739851, 32.7200508, -44.2395592, 41.4989662
4: -12.2073965, 26.5305767, -14.5391541, 32.1684265, -44.3758125, 41.0697327

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.2679539, 23.6466656, -7.8287406, 28.8187637, -35.0867157, 31.4754066
1: -7.2751541, 26.7506466, -9.1962099, 32.5476151, -39.8227692, 35.9468575
2: -7.7640185, 26.8291969, -9.6979246, 32.8125992, -40.5766182, 36.5271149
3: -11.5195141, 27.4249935, -14.3933182, 33.4870682, -45.0065727, 41.8183060
4: -12.2073965, 26.5305767, -14.8758755, 32.8492126, -45.0566025, 41.4064445

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.6865768, 32.5887604, -7.6489487, 28.1001740, -36.7867317, 40.2377052
1: -10.2309980, 37.0391312, -8.9741259, 31.7052670, -41.9362640, 46.0132484
2: -10.7693300, 37.0428123, -9.4762421, 32.0012550, -42.7705803, 46.5190544
3: -16.0634308, 38.2776337, -14.0308552, 32.5917473, -48.6551781, 52.3084869
4: -17.0398197, 36.4654579, -14.4800863, 32.0801010, -49.1199188, 50.9455452

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.6865768, 32.5887604, -7.8041110, 28.7063828, -37.3929558, 40.3928719
1: -10.2309980, 37.0391312, -9.1664591, 32.4109993, -42.6419868, 46.2055893
2: -10.7693300, 37.0428123, -9.6695223, 32.6869049, -43.4562340, 46.7123337
3: -16.0634308, 38.2776337, -14.3423948, 33.3406715, -49.4041023, 52.6200294
4: -17.0398197, 36.4654579, -14.8098640, 32.7422562, -49.7820740, 51.2753220

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.08 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9647384, 22.4902878, -6.1583133, 23.1884537, -29.1531906, 28.6486015
1: -6.9048386, 25.4210529, -7.1370277, 26.2245445, -33.1293755, 32.5580788
2: -7.3848820, 25.5047226, -7.6254220, 26.3030167, -33.6878967, 33.1301384
3: -10.9337454, 26.0330429, -11.2898073, 26.8580627, -37.7918015, 37.3228455
4: -11.5946884, 25.2559605, -11.9580927, 26.0414906, -37.6361771, 37.2140503

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5688609, upper bound: 43.5588921
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5687841, upper bound: 43.5588165
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.3492012, 31.2999210, -6.1583133, 23.1884537, -31.5376549, 37.4582291
1: -9.8115788, 35.5474586, -7.1370277, 26.2245445, -36.0361214, 42.6844826
2: -10.3486872, 35.5690842, -7.6254220, 26.3030167, -36.6517029, 43.1945076
3: -15.4028683, 36.7234039, -11.2898073, 26.8580627, -42.2609253, 48.0132065
4: -16.3597260, 35.0432816, -11.9580927, 26.0414906, -42.4012146, 47.0013733

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5688609, upper bound: 43.5588921
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5687841, upper bound: 43.5588165
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9565649, 22.4609966, -8.5555115, 32.0344582, -37.9910202, 31.0165081
1: -6.8949361, 25.3889275, -10.0620441, 36.3880119, -43.2829475, 35.4509735
2: -7.3748474, 25.4706459, -10.6037455, 36.4118919, -43.7867393, 36.0743904
3: -10.9184370, 25.9993172, -15.7867413, 37.5937805, -48.5122147, 41.7860489
4: -11.5800266, 25.2204456, -16.7442265, 35.8790398, -47.4590683, 41.9646645

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.2806950, 31.0336685, -8.5555115, 32.0344582, -40.3151436, 39.5891800
1: -9.6846685, 35.2271957, -10.0620441, 36.3880119, -46.0726814, 45.2892380
2: -10.2633362, 35.2591934, -10.6037455, 36.4118919, -46.6752243, 45.8629379
3: -15.2045250, 36.3579483, -15.7867413, 37.5937805, -52.7983055, 52.1446838
4: -16.2140293, 34.7246552, -16.7442265, 35.8790398, -52.0930710, 51.4688797

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.0948410, 22.9826908, -29.4889946, 30.3624763
1: -7.5842962, 27.4015598, -7.0591693, 26.0043926, -33.5886841, 34.4607277
2: -8.0552320, 27.5818462, -7.5463943, 26.0642319, -34.1194649, 35.1282387
3: -11.9514847, 28.1189060, -11.1737213, 26.6180305, -38.5695076, 39.2926216
4: -12.5255804, 27.4433079, -11.8461428, 25.7902393, -38.3158112, 39.2894516

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579472, upper bound: 43.5585293
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579165, upper bound: 43.5584552
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.0948410, 22.9826908, -31.9075127, 39.2801895
1: -10.5382071, 37.6687279, -7.0591693, 26.0043926, -36.5425949, 44.7278976
2: -11.0602732, 37.7767830, -7.5463943, 26.0642319, -37.1245041, 45.3231773
3: -16.4940701, 38.9501419, -11.1737213, 26.6180305, -43.1120949, 50.1238594
4: -17.3545589, 37.3720055, -11.8461428, 25.7902393, -43.1447945, 49.2181473

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579472, upper bound: 43.5585293
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579165, upper bound: 43.5584552
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.4854784, 31.8138542, -38.3201561, 32.7531166
1: -7.5842962, 27.4015598, -9.9755898, 36.1475639, -43.7318573, 37.3771477
2: -8.0552320, 27.5818462, -10.5169134, 36.1537857, -44.2090187, 38.0987511
3: -11.9514847, 28.1189060, -15.6582737, 37.3285637, -49.2800484, 43.7771759
4: -12.5255804, 27.4433079, -16.6212826, 35.6058006, -48.1313820, 44.0645905

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.4854784, 31.8138542, -40.7386742, 41.6708298
1: -10.5382071, 37.6687279, -9.9755898, 36.1475639, -46.6857719, 47.6443176
2: -11.0602732, 37.7767830, -10.5169134, 36.1537857, -47.2140579, 48.2936974
3: -16.4940701, 38.9501419, -15.6582737, 37.3285637, -53.8226318, 54.6084137
4: -17.3545589, 37.3720055, -16.6212826, 35.6058006, -52.9603577, 53.9932861

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9647384, 22.4902878, -6.3281989, 23.8465290, -29.8112679, 28.8184872
1: -6.9048386, 25.4210529, -7.3499689, 26.9663219, -33.8711548, 32.7710228
2: -7.3848820, 25.5047226, -7.8391385, 27.0614948, -34.4463768, 33.3438568
3: -10.9337454, 26.0330429, -11.6318560, 27.6643848, -38.5981255, 37.6648979
4: -11.5946884, 25.2559605, -12.3158455, 26.7735996, -38.3682823, 37.5718040

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5394532, upper bound: 43.5222790
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5376647, upper bound: 43.5208341
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.3492012, 31.2999210, -6.3281989, 23.8465290, -32.1957245, 37.6281204
1: -9.8115788, 35.5474586, -7.3499689, 26.9663219, -36.7778931, 42.8974266
2: -10.3486872, 35.5690842, -7.8391385, 27.0614948, -37.4101830, 43.4082184
3: -15.4028683, 36.7234039, -11.6318560, 27.6643848, -43.0672531, 48.3552589
4: -16.3597260, 35.0432816, -12.3158455, 26.7735996, -43.1333237, 47.3591270

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5394532, upper bound: 43.5222790
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5376647, upper bound: 43.5208341
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9565649, 22.4609966, -8.7499533, 32.7939873, -38.7505531, 31.2109470
1: -6.8949361, 25.3889275, -10.3097925, 37.2698402, -44.1647758, 35.6987190
2: -7.3748474, 25.4706459, -10.8480654, 37.2827568, -44.6576042, 36.3187027
3: -10.9184370, 25.9993172, -16.1813354, 38.5235176, -49.4419518, 42.1806526
4: -11.5800266, 25.2204456, -17.1520309, 36.7178192, -48.2978439, 42.3724709

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.2806950, 31.0336685, -8.7499533, 32.7939873, -41.0746765, 39.7836189
1: -9.6846685, 35.2271957, -10.3097925, 37.2698402, -46.9545097, 45.5369873
2: -10.2633362, 35.2591934, -10.8480654, 37.2827568, -47.5460930, 46.1072502
3: -15.2045250, 36.3579483, -16.1813354, 38.5235176, -53.7280426, 52.5392838
4: -16.2140293, 34.7246552, -17.1520309, 36.7178192, -52.9318466, 51.8766861

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -6.2679539, 23.6466656, -30.1529694, 30.5355911
1: -7.5842962, 27.4015598, -7.2751541, 26.7506466, -34.3349419, 34.6767120
2: -8.0552320, 27.5818462, -7.7640185, 26.8291969, -34.8844299, 35.3458633
3: -11.9514847, 28.1189060, -11.5195141, 27.4249935, -39.3764725, 39.6384163
4: -12.5255804, 27.4433079, -12.2073965, 26.5305767, -39.0561562, 39.6506958

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5222874
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5354970, upper bound: 43.5208401
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -6.2679539, 23.6466656, -32.5714874, 39.4533081
1: -10.5382071, 37.6687279, -7.2751541, 26.7506466, -37.2888527, 44.9438820
2: -11.0602732, 37.7767830, -7.7640185, 26.8291969, -37.8894691, 45.5408020
3: -16.4940701, 38.9501419, -11.5195141, 27.4249935, -43.9190598, 50.4696503
4: -17.3545589, 37.3720055, -12.2073965, 26.5305767, -43.8851357, 49.5793915

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5222874
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5354970, upper bound: 43.5208401
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.6865768, 32.5887604, -39.0950623, 32.9542122
1: -7.5842962, 27.4015598, -10.2309980, 37.0391312, -44.6234245, 37.6325531
2: -8.0552320, 27.5818462, -10.7693300, 37.0428123, -45.0980453, 38.3511734
3: -11.9514847, 28.1189060, -16.0634308, 38.2776337, -50.2291183, 44.1823349
4: -12.5255804, 27.4433079, -17.0398197, 36.4654579, -48.9910393, 44.4831276

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.6865768, 32.5887604, -41.5135803, 41.8719139
1: -10.5382071, 37.6687279, -10.2309980, 37.0391312, -47.5773354, 47.8997269
2: -11.0602732, 37.7767830, -10.7693300, 37.0428123, -48.1030846, 48.5461121
3: -16.4940701, 38.9501419, -16.0634308, 38.2776337, -54.7717056, 55.0135727
4: -17.3545589, 37.3720055, -17.0398197, 36.4654579, -53.8200150, 54.4118233

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9042511, 22.3964958, -6.9936957, 25.9564800, -31.8607311, 29.3901920
1: -6.8388619, 25.3658218, -8.1467400, 29.3088837, -36.1477432, 33.5125618
2: -7.3123736, 25.3934631, -8.6618214, 29.4827328, -36.7950974, 34.0552826
3: -10.8580732, 25.9775810, -12.7979937, 30.0797024, -40.9377747, 38.7755737
4: -11.5812035, 25.0183811, -13.3646317, 29.4214401, -41.0026436, 38.3830032

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222790, upper bound: 43.5394532
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222790, upper bound: 43.5394532
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.4716568, 24.4816589, -7.0731578, 26.2191887, -32.6908455, 31.5548172
1: -7.5054522, 27.7502098, -8.2421904, 29.6049995, -37.1104507, 35.9923935
2: -8.0167580, 27.7686310, -8.7603493, 29.7844105, -37.8011703, 36.5289803
3: -11.8870697, 28.4665527, -12.9405422, 30.3858204, -42.2728882, 41.4070892
4: -12.6870508, 27.3353367, -13.4987841, 29.7390289, -42.4260788, 40.8341217

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208341, upper bound: 43.5376647
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208341, upper bound: 43.5376647
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9042511, 22.3964958, -7.1749377, 26.6678543, -32.5721054, 29.5714321
1: -6.8388619, 25.3658218, -8.3748884, 30.1260242, -36.9648857, 33.7407074
2: -7.3123736, 25.3934631, -8.8888168, 30.3033009, -37.6156731, 34.2822800
3: -10.8580732, 25.9775810, -13.1653433, 30.9486065, -41.8066788, 39.1429214
4: -11.5812035, 25.0183811, -13.7452297, 30.2123890, -41.7935867, 38.7636070

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.4716568, 24.4816589, -7.2462487, 26.8973961, -33.3690529, 31.7279072
1: -7.5054522, 27.7502098, -8.4596024, 30.3843517, -37.8898048, 36.2098122
2: -8.0167580, 27.7686310, -8.9771776, 30.5662880, -38.5830460, 36.7458038
3: -11.8870697, 28.4665527, -13.2913275, 31.2153854, -43.1024551, 41.7578735
4: -12.6870508, 27.3353367, -13.8636837, 30.4928055, -43.1798515, 41.1990204

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -7.0997167, 26.2901516, -34.8340759, 39.1628304
1: -10.0602074, 36.4324837, -8.2793884, 29.6757069, -39.7359161, 44.7118721
2: -10.5933409, 36.4442482, -8.7958736, 29.8768692, -40.4702034, 45.2401199
3: -15.7987118, 37.6562881, -12.9914083, 30.4564915, -46.2551918, 50.6476936
4: -16.7670708, 35.8889656, -13.5233612, 29.8568954, -46.6239662, 49.4123268

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.0997167, 26.2901516, -35.4017334, 41.0122566
1: -10.7727814, 38.5158424, -8.2793884, 29.6757069, -40.4484863, 46.7952271
2: -11.2928276, 38.6079216, -8.7958736, 29.8768692, -41.1696968, 47.4037933
3: -16.8676147, 39.8401985, -12.9914083, 30.4564915, -47.3240967, 52.8316040
4: -17.7444897, 38.1677361, -13.5233612, 29.8568954, -47.6013870, 51.6910973

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -7.2589397, 26.9156380, -35.4595642, 39.3220482
1: -10.0602074, 36.4324837, -8.4761724, 30.3951454, -40.4553528, 44.9086571
2: -10.5933409, 36.4442482, -8.9955788, 30.5927124, -41.1860542, 45.4398193
3: -15.7987118, 37.6562881, -13.3101597, 31.2216206, -47.0203323, 50.9664459
4: -16.7670708, 35.8889656, -13.8618517, 30.5465946, -47.3136673, 49.7508163

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.2589397, 26.9156380, -36.0272255, 41.1714745
1: -10.7727814, 38.5158424, -8.4761724, 30.3951454, -41.1679268, 46.9920158
2: -11.2928276, 38.6079216, -8.9955788, 30.5927124, -41.8855400, 47.6034927
3: -16.8676147, 39.8401985, -13.3101597, 31.2216206, -48.0892334, 53.1503601
4: -17.7444897, 38.1677361, -13.8618517, 30.5465946, -48.2910843, 52.0295868

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.8455057, 22.1999588, -7.5353861, 27.7428322, -33.5883369, 29.7353439
1: -6.7665153, 25.1566353, -8.8346891, 31.3137150, -38.0802193, 33.9913254
2: -7.2386465, 25.1651039, -9.3335552, 31.5802689, -38.8189163, 34.4986572
3: -10.7506304, 25.7549343, -13.8271599, 32.1893425, -42.9399719, 39.5820808
4: -11.4763651, 24.7800026, -14.3069239, 31.6180553, -43.0944214, 39.0869255

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222790, upper bound: 43.5371676
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222874, upper bound: 43.5371676
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.4144363, 24.2925529, -7.6128840, 27.9990616, -34.4134979, 31.9054356
1: -7.4344707, 27.5488415, -8.9273090, 31.6022205, -39.0366898, 36.4761505
2: -7.9455099, 27.5484562, -9.4298067, 31.8748436, -39.8203430, 36.9782639
3: -11.7801027, 28.2418938, -13.9653015, 32.4874916, -44.2675934, 42.2071915
4: -12.5833654, 27.1059551, -14.4383240, 31.9280090, -44.5113754, 41.5442772

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208401, upper bound: 43.5354970
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208401, upper bound: 43.5354970
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.8455057, 22.1999588, -7.6925006, 28.3592300, -34.2047348, 29.8924599
1: -6.7665153, 25.1566353, -9.0298738, 32.0308723, -38.7973862, 34.1865005
2: -7.2386465, 25.1651039, -9.5294046, 32.2814789, -39.5201263, 34.6945038
3: -10.7506304, 25.7549343, -14.1426992, 32.9495811, -43.7002106, 39.8976288
4: -11.4763651, 24.7800026, -14.6405544, 32.2928581, -43.7692146, 39.4205551

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.4144363, 24.2925529, -7.7637253, 28.5871029, -35.0015373, 32.0562744
1: -7.4344707, 27.5488415, -9.1141834, 32.2879333, -39.7224007, 36.6630249
2: -7.9455099, 27.5484562, -9.6177616, 32.5425568, -40.4880676, 37.1662178
3: -11.7801027, 28.2418938, -14.2680349, 33.2154236, -44.9955254, 42.5099258
4: -12.5833654, 27.1059551, -14.7597780, 32.5705414, -45.1539078, 41.8657341

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -7.6489487, 28.1001740, -36.6440926, 39.7120552
1: -10.0602074, 36.4324837, -8.9741259, 31.7052670, -41.7654724, 45.4066010
2: -10.5933409, 36.4442482, -9.4762421, 32.0012550, -42.5945892, 45.9204903
3: -15.7987118, 37.6562881, -14.0308552, 32.5917473, -48.3904572, 51.6871414
4: -16.7670708, 35.8889656, -14.4800863, 32.0801010, -48.8471718, 50.3690529

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.6489487, 28.1001740, -37.2117538, 41.5614853
1: -10.7727814, 38.5158424, -8.9741259, 31.7052670, -42.4780502, 47.4899597
2: -11.2928276, 38.6079216, -9.4762421, 32.0012550, -43.2940826, 48.0841637
3: -16.8676147, 39.8401985, -14.0308552, 32.5917473, -49.4593620, 53.8710556
4: -17.7444897, 38.1677361, -14.4800863, 32.0801010, -49.8245926, 52.6478233

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -7.8041110, 28.7063828, -37.2503090, 39.8672218
1: -10.0602074, 36.4324837, -9.1664591, 32.4109993, -42.4712029, 45.5989418
2: -10.5933409, 36.4442482, -9.6695223, 32.6869049, -43.2802429, 46.1137695
3: -15.7987118, 37.6562881, -14.3423948, 33.3406715, -49.1393814, 51.9986839
4: -16.7670708, 35.8889656, -14.8098640, 32.7422562, -49.5093269, 50.6988297

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -7.8041110, 28.7063828, -37.8179703, 41.7166519
1: -10.7727814, 38.5158424, -9.1664591, 32.4109993, -43.1837807, 47.6823006
2: -11.2928276, 38.6079216, -9.6695223, 32.6869049, -43.9797325, 48.2774429
3: -16.8676147, 39.8401985, -14.3423948, 33.3406715, -50.2082863, 54.1825943
4: -17.7444897, 38.1677361, -14.8098640, 32.7422562, -50.4867477, 52.9776001

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.36 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5688609, upper bound: 43.5588921
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5687841, upper bound: 43.5588165
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5688609, upper bound: 43.5588921
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5687841, upper bound: 43.5588165
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5673765, upper bound: 43.5568837
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5579472, upper bound: 43.5585293
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5579165, upper bound: 43.5584552
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5579472, upper bound: 43.5585293
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5579165, upper bound: 43.5584552
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5565484, upper bound: 43.5565484
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5394532, upper bound: 43.5222790
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5376647, upper bound: 43.5208341
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5394532, upper bound: 43.5222790
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5376647, upper bound: 43.5208341
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5326418, upper bound: 43.5161731
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5222874
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5354970, upper bound: 43.5208401
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5222874
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5354970, upper bound: 43.5208401
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5306355, upper bound: 43.5161823
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5222790, upper bound: 43.5394532
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5222790, upper bound: 43.5394532
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5208341, upper bound: 43.5376647
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5208341, upper bound: 43.5376647
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5208220, upper bound: 43.5376089
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5194948, upper bound: 43.5361327
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5326418
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5149802, upper bound: 43.5315735
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5222790, upper bound: 43.5371676
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5222874, upper bound: 43.5371676
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5208401, upper bound: 43.5354970
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5208401, upper bound: 43.5354970
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5191840, upper bound: 43.5168084
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5179210, upper bound: 43.5163758
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5161731, upper bound: 43.5306355
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 3, lower bound: -43.5134444, upper bound: 43.5134444

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.8418798, 22.0709801, -5.7395577, 21.7566948, -27.5985718, 27.8105373
1: -6.7569051, 24.9581604, -6.6321549, 24.6417408, -31.3986454, 31.5903149
2: -7.2322412, 25.0225143, -7.1045218, 24.6530190, -31.8852596, 32.1270370
3: -10.7100267, 25.5454636, -10.5278606, 25.2069702, -35.9169960, 36.0733185
4: -11.3829393, 24.7480869, -11.2362003, 24.3051052, -35.6880341, 35.9842873

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5763326, upper bound: 43.5610785
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5763326, upper bound: 43.5610785
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.8938942, 22.2443905, -6.3001671, 23.8188057, -29.7126961, 28.5445576
1: -6.8175750, 25.1503067, -7.2893491, 27.0015850, -33.8191605, 32.4396553
2: -7.2974815, 25.2184639, -7.8004341, 27.0059090, -34.3033791, 33.0188980
3: -10.8006182, 25.7475243, -11.5409155, 27.6599293, -38.4605484, 37.2884407
4: -11.4699802, 24.9590034, -12.3259583, 26.5960751, -38.0660477, 37.2849579

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5763326, upper bound: 43.5610785
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5763326, upper bound: 43.5610785
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.2165375, 30.8563671, -5.7395577, 21.7566948, -29.9732304, 36.5959244
1: -9.6504583, 35.0517197, -6.6321549, 24.6417408, -34.2921982, 41.6838722
2: -10.1851196, 35.0528946, -7.1045218, 24.6530190, -34.8381386, 42.1574173
3: -15.1598892, 36.2057915, -10.5278606, 25.2069702, -40.3668594, 46.7336464
4: -16.1322193, 34.4989471, -11.2362003, 24.3051052, -40.4373207, 45.7351456

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5393349, upper bound: 43.5223857
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5246957, upper bound: 43.5204681
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.2939920, 31.1040878, -6.3001671, 23.8188057, -32.1127968, 37.4042549
1: -9.7430067, 35.3283005, -7.2893491, 27.0015850, -36.7445908, 42.6176491
2: -10.2804527, 35.3408737, -7.8004341, 27.0059090, -37.2863464, 43.1413078
3: -15.2978373, 36.4949608, -11.5409155, 27.6599293, -42.9577637, 48.0358772
4: -16.2612801, 34.8064117, -12.3259583, 26.5960751, -42.8573456, 47.1323662

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5393349, upper bound: 43.5247104
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5246366, upper bound: 43.5222374
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9565649, 22.4609966, -8.3492012, 31.2999210, -37.2564850, 30.8101978
1: -6.8949361, 25.3889275, -9.8115788, 35.5474586, -42.4423904, 35.2005081
2: -7.3748474, 25.4706459, -10.3486872, 35.5690842, -42.9439316, 35.8193321
3: -10.9184370, 25.9993172, -15.4028683, 36.7234039, -47.6418381, 41.4021835
4: -11.5800266, 25.2204456, -16.3597260, 35.0432816, -46.6233063, 41.5801659

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5771248, upper bound: 43.5595030
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5747084, upper bound: 43.5589605
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9565649, 22.4609966, -8.9248209, 33.1853523, -39.1419144, 31.3858166
1: -6.8949361, 25.3889275, -10.5382071, 37.6687279, -44.5636635, 35.9271317
2: -7.3748474, 25.4706459, -11.0602732, 37.7767830, -45.1516304, 36.5309181
3: -10.9184370, 25.9993172, -16.4940701, 38.9501419, -49.8685760, 42.4933853
4: -11.5800266, 25.2204456, -17.3545589, 37.3720055, -48.9520302, 42.5750046

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5771248, upper bound: 43.5595030
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5747084, upper bound: 43.5589605
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.2806950, 31.0336685, -8.3492012, 31.2999210, -39.5806084, 39.3828659
1: -9.6846685, 35.2271957, -9.8115788, 35.5474586, -45.2321243, 45.0387726
2: -10.2633362, 35.2591934, -10.3486872, 35.5690842, -45.8324203, 45.6078796
3: -15.2045250, 36.3579483, -15.4028683, 36.7234039, -51.9279289, 51.7608109
4: -16.2140293, 34.7246552, -16.3597260, 35.0432816, -51.2573090, 51.0843811

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5411955, upper bound: 43.5247104
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5227563, upper bound: 43.5222374
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.2806950, 31.0336685, -8.9248209, 33.1853523, -41.4660339, 39.9584885
1: -9.6846685, 35.2271957, -10.5382071, 37.6687279, -47.3533974, 45.7653999
2: -10.2633362, 35.2591934, -11.0602732, 37.7767830, -48.0401192, 46.3194656
3: -15.2045250, 36.3579483, -16.4940701, 38.9501419, -54.1546669, 52.8520164
4: -16.2140293, 34.7246552, -17.3545589, 37.3720055, -53.5860329, 52.0792160

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5411955, upper bound: 43.5247104
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5227563, upper bound: 43.5222374
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.3812270, 23.8414116, -5.6801596, 21.5670872, -27.9483128, 29.5215721
1: -7.4330468, 26.9280090, -6.5579309, 24.4299488, -31.8629951, 33.4859390
2: -7.8993645, 27.0912895, -7.0303006, 24.4215813, -32.3209457, 34.1215820
3: -11.7228756, 27.6214695, -10.4185915, 24.9845695, -36.7074432, 38.0400543
4: -12.3090363, 26.9261112, -11.1333103, 24.0630283, -36.3720589, 38.0594215

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5336598, upper bound: 43.5561739
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5600489, upper bound: 43.5600833
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.4324541, 24.0112610, -6.2318587, 23.5979729, -30.0304222, 30.2431202
1: -7.4915299, 27.1158504, -7.2049680, 26.7609234, -34.2524529, 34.3208199
2: -7.9639883, 27.2831879, -7.7155862, 26.7438545, -34.7078362, 34.9987755
3: -11.8102016, 27.8177853, -11.4157734, 27.4039516, -39.2141533, 39.2335510
4: -12.3942461, 27.1339741, -12.2057104, 26.3229675, -38.7172127, 39.3396835

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5335787, upper bound: 43.5561266
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5600086, upper bound: 43.5600086
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.7922382, 32.7415848, -5.6801596, 21.5670872, -30.3593216, 38.4217453
1: -10.3767853, 37.1724854, -6.5579309, 24.4299488, -34.8067322, 43.7304153
2: -10.8961630, 37.2604828, -7.0303006, 24.4215813, -35.3177452, 44.2907791
3: -16.2511063, 38.4314346, -10.4185915, 24.9845695, -41.2356720, 48.8500214
4: -17.1261673, 36.8279686, -11.1333103, 24.0630283, -41.1891823, 47.9612808

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5369828, upper bound: 43.5223933
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5241771, upper bound: 43.5205593
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8665609, 32.9784851, -6.2318587, 23.5979729, -32.4645309, 39.2103424
1: -10.4655848, 37.4369583, -7.2049680, 26.7609234, -37.2265091, 44.6419258
2: -10.9879742, 37.5357208, -7.7155862, 26.7438545, -37.7318230, 45.2513084
3: -16.3830986, 38.7082291, -11.4157734, 27.4039516, -43.7870483, 50.1239967
4: -17.2503986, 37.1225052, -12.2057104, 26.3229675, -43.5733643, 49.3282166

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5401904, upper bound: 43.5247142
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5241180, upper bound: 43.5223355
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.3492012, 31.2999210, -37.8062248, 32.6168404
1: -7.5842962, 27.4015598, -9.8115788, 35.5474586, -43.1317520, 37.2131348
2: -8.0552320, 27.5818462, -10.3486872, 35.5690842, -43.6243172, 37.9305344
3: -11.9514847, 28.1189060, -15.4028683, 36.7234039, -48.6748848, 43.5217743
4: -12.5255804, 27.4433079, -16.3597260, 35.0432816, -47.5688629, 43.8030281

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585293, upper bound: 43.5579472
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5584552, upper bound: 43.5579165
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.5063033, 24.2676392, -8.9248209, 33.1853523, -39.6916504, 33.1924591
1: -7.5842962, 27.4015598, -10.5382071, 37.6687279, -45.2530251, 37.9397621
2: -8.0552320, 27.5818462, -11.0602732, 37.7767830, -45.8320160, 38.6421165
3: -11.9514847, 28.1189060, -16.4940701, 38.9501419, -50.9016266, 44.6129761
4: -12.5255804, 27.4433079, -17.3545589, 37.3720055, -49.8975830, 44.7978668

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585293, upper bound: 43.5579472
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5584552, upper bound: 43.5579165
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9248209, 33.1853523, -8.3492012, 31.2999210, -40.2247429, 41.5345459
1: -10.5382071, 37.6687279, -9.8115788, 35.5474586, -46.0856628, 47.4803085
2: -11.0602732, 37.7767830, -10.3486872, 35.5690842, -46.6293564, 48.1254692
3: -16.4940701, 38.9501419, -15.4028683, 36.7234039, -53.2174759, 54.3530083
4: -17.3545589, 37.3720055, -16.3597260, 35.0432816, -52.3978424, 53.7317276

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1133.27 seconds
