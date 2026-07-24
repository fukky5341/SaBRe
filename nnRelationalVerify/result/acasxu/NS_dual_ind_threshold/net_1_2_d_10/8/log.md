## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 43.3827531155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307)
1: (-8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275)
2: (-9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216)
3: (-14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580)
4: (-14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.42 = 2.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5965478, upper bound: 43.5783800
time: 0.40 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 3, lower bound: -43.5965478, upper bound: 43.5783800
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.4532280, 27.5705853, -7.6606002, 28.3044319, -35.7576599, 35.2311859
1: -8.7061062, 31.1410255, -8.9601765, 31.9744511, -40.6805573, 40.1012039
2: -9.2308626, 31.3365688, -9.4893942, 32.1866379, -41.4174995, 40.8259621
3: -13.6493845, 31.9803791, -14.0368652, 32.8534927, -46.5028763, 46.0172424
4: -14.1980610, 31.2951145, -14.5853920, 32.1437492, -46.3418121, 45.8805046

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.45 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.6380959, 28.2877064, -7.6606002, 28.3044319, -35.9425278, 35.9483070
1: -8.9388247, 31.9677963, -8.9601765, 31.9744511, -40.9132729, 40.9279709
2: -9.4615984, 32.1662331, -9.4893942, 32.1866379, -41.6482315, 41.6556206
3: -14.0235004, 32.8585358, -14.0368652, 32.8534927, -46.8769913, 46.8954010
4: -14.5844440, 32.0941963, -14.5853920, 32.1437492, -46.7281952, 46.6795807

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.38 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
time: 0.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -43.5760378, upper bound: 43.5760378

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.4532280, 27.5705853, -7.4532280, 27.5705853, -35.0238037, 35.0238037
1: -8.7061062, 31.1410255, -8.7061062, 31.1410255, -39.8471298, 39.8471298
2: -9.2308626, 31.3365688, -9.2308626, 31.3365688, -40.5674324, 40.5674324
3: -13.6493845, 31.9803791, -13.6493845, 31.9803791, -45.6297646, 45.6297646
4: -14.1980610, 31.2951145, -14.1980610, 31.2951145, -45.4931755, 45.4931755

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5915466, upper bound: 43.5325931
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.40 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.4532280, 27.5705853, -7.6380959, 28.2877064, -35.7409325, 35.2086716
1: -8.7061062, 31.1410255, -8.9388247, 31.9677963, -40.6739006, 40.0798416
2: -9.2308626, 31.3365688, -9.4615984, 32.1662331, -41.3970947, 40.7981682
3: -13.6493845, 31.9803791, -14.0235004, 32.8585358, -46.5079193, 46.0038795
4: -14.1980610, 31.2951145, -14.5844440, 32.0941963, -46.2922554, 45.8795586

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5915466, upper bound: 43.5325931
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.76 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.6380959, 28.2877064, -7.4532280, 27.5705853, -35.2086716, 35.7409363
1: -8.9388247, 31.9677963, -8.7061062, 31.1410255, -40.0798416, 40.6739044
2: -9.4615984, 32.1662331, -9.2308626, 31.3365688, -40.7981682, 41.3970947
3: -14.0235004, 32.8585358, -13.6493845, 31.9803791, -46.0038795, 46.5079193
4: -14.5844440, 32.0941963, -14.1980610, 31.2951145, -45.8795586, 46.2922554

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.43 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.6380959, 28.2877064, -7.6380959, 28.2877064, -35.9258003, 35.9258003
1: -8.9388247, 31.9677963, -8.9388247, 31.9677963, -40.9066124, 40.9066124
2: -9.4615984, 32.1662331, -9.4615984, 32.1662331, -41.6278305, 41.6278305
3: -14.0235004, 32.8585358, -14.0235004, 32.8585358, -46.8820343, 46.8820343
4: -14.5844440, 32.0941963, -14.5844440, 32.0941963, -46.6786385, 46.6786385

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.44 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5915466, upper bound: 43.5325931
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5915466, upper bound: 43.5325931
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5713141, upper bound: 43.5302640
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.1871862, 26.6099091, -7.4532280, 27.5705853, -34.7577667, 34.0631332
1: -8.3826809, 30.0424767, -8.7061062, 31.1410255, -39.5237045, 38.7485809
2: -8.9014492, 30.2400932, -9.2308626, 31.3365688, -40.2380180, 39.4709549
3: -13.1529007, 30.8414726, -13.6493845, 31.9803791, -45.1332779, 44.4908562
4: -13.6967449, 30.2116623, -14.1980610, 31.2951145, -44.9918594, 44.4097214

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.7295799, 28.3989639, -7.4532280, 27.5705853, -35.3001633, 35.8521881
1: -9.0717134, 32.0508652, -8.7061062, 31.1410255, -40.2127304, 40.7569733
2: -9.5743036, 32.3409538, -9.2308626, 31.3365688, -40.9108734, 41.5718155
3: -14.1834040, 32.9562950, -13.6493845, 31.9803791, -46.1637726, 46.6056786
4: -14.6434193, 32.4114075, -14.1980610, 31.2951145, -45.9385338, 46.6094666

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.1871862, 26.6099091, -7.6380959, 28.2877064, -35.4748917, 34.2480011
1: -8.3826809, 30.0424767, -8.9388247, 31.9677963, -40.3504791, 38.9812927
2: -8.9014492, 30.2400932, -9.4615984, 32.1662331, -41.0676804, 39.7016907
3: -13.1529007, 30.8414726, -14.0235004, 32.8585358, -46.0114365, 44.8649750
4: -13.6967449, 30.2116623, -14.5844440, 32.0941963, -45.7909393, 44.7961044

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.7295799, 28.3989639, -7.6380959, 28.2877064, -36.0172882, 36.0370560
1: -9.0717134, 32.0508652, -8.9388247, 31.9677963, -41.0395050, 40.9896851
2: -9.5743036, 32.3409538, -9.4615984, 32.1662331, -41.7405319, 41.8025513
3: -14.1834040, 32.9562950, -14.0235004, 32.8585358, -47.0419350, 46.9797974
4: -14.6434193, 32.4114075, -14.5844440, 32.0941963, -46.7376099, 46.9958496

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.3701210, 27.3269844, -7.4532280, 27.5705853, -34.9407043, 34.7802124
1: -8.6132603, 30.8662567, -8.7061062, 31.1410255, -39.7542839, 39.5723572
2: -9.1305141, 31.0644073, -9.2308626, 31.3365688, -40.4670830, 40.2952690
3: -13.5239582, 31.7180500, -13.6493845, 31.9803791, -45.5043373, 45.3674355
4: -14.0812511, 31.0096245, -14.1980610, 31.2951145, -45.3763618, 45.2076836

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.9041195, 29.0763073, -7.4532280, 27.5705853, -35.4747047, 36.5295334
1: -9.2882051, 32.8371925, -8.7061062, 31.1410255, -40.4292297, 41.5432930
2: -9.7911968, 33.1104507, -9.2308626, 31.3365688, -41.1277657, 42.3413124
3: -14.5323257, 33.7886276, -13.6493845, 31.9803791, -46.5126991, 47.4380112
4: -15.0085173, 33.1586647, -14.1980610, 31.2951145, -46.3036270, 47.3567276

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
time: 0.39 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3701210, 27.3269844, -7.6380959, 28.2877064, -35.6578293, 34.9650764
1: -8.6132603, 30.8662567, -8.9388247, 31.9677963, -40.5810547, 39.8050690
2: -9.1305141, 31.0644073, -9.4615984, 32.1662331, -41.2967453, 40.5260048
3: -13.5239582, 31.7180500, -14.0235004, 32.8585358, -46.3824921, 45.7415504
4: -14.0812511, 31.0096245, -14.5844440, 32.0941963, -46.1754379, 45.5940704

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.9041195, 29.0763073, -7.6380959, 28.2877064, -36.1918259, 36.7144012
1: -9.2882051, 32.8371925, -8.9388247, 31.9677963, -41.2560005, 41.7760048
2: -9.7911968, 33.1104507, -9.4615984, 32.1662331, -41.9574280, 42.5720482
3: -14.5323257, 33.7886276, -14.0235004, 32.8585358, -47.3908577, 47.8121262
4: -15.0085173, 33.1586647, -14.5844440, 32.0941963, -47.1027031, 47.7431107

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.44 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
time: 0.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.42 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5611934, upper bound: 43.5611934
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5588106, upper bound: 43.5279232
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5279232, upper bound: 43.5588106
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.5255404, upper bound: 43.5255404

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.1871862, 26.6099091, -7.1871862, 26.6099091, -33.7970886, 33.7970886
1: -8.3826809, 30.0424767, -8.3826809, 30.0424767, -38.4251556, 38.4251556
2: -8.9014492, 30.2400932, -8.9014492, 30.2400932, -39.1415405, 39.1415405
3: -13.1529007, 30.8414726, -13.1529007, 30.8414726, -43.9943733, 43.9943733
4: -13.6967449, 30.2116623, -13.6967449, 30.2116623, -43.9084091, 43.9084091

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.1871862, 26.6099091, -7.7295799, 28.3989639, -35.5861473, 34.3394890
1: -8.3826809, 30.0424767, -9.0717134, 32.0508652, -40.4335480, 39.1141815
2: -8.9014492, 30.2400932, -9.5743036, 32.3409538, -41.2424011, 39.8143959
3: -13.1529007, 30.8414726, -14.1834040, 32.9562950, -46.1091957, 45.0248756
4: -13.6967449, 30.2116623, -14.6434193, 32.4114075, -46.1081543, 44.8550797

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.7295799, 28.3989639, -7.1871862, 26.6099091, -34.3394890, 35.5861473
1: -9.0717134, 32.0508652, -8.3826809, 30.0424767, -39.1141815, 40.4335480
2: -9.5743036, 32.3409538, -8.9014492, 30.2400932, -39.8143959, 41.2424011
3: -14.1834040, 32.9562950, -13.1529007, 30.8414726, -45.0248756, 46.1091957
4: -14.6434193, 32.4114075, -13.6967449, 30.2116623, -44.8550797, 46.1081543

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.7295799, 28.3989639, -7.7295799, 28.3989639, -36.1285439, 36.1285439
1: -9.0717134, 32.0508652, -9.0717134, 32.0508652, -41.1225739, 41.1225739
2: -9.5743036, 32.3409538, -9.5743036, 32.3409538, -41.9152565, 41.9152565
3: -14.1834040, 32.9562950, -14.1834040, 32.9562950, -47.1396904, 47.1396904
4: -14.6434193, 32.4114075, -14.6434193, 32.4114075, -47.0548248, 47.0548248

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.1871862, 26.6099091, -7.3701210, 27.3269844, -34.5141716, 33.9800301
1: -8.3826809, 30.0424767, -8.6132603, 30.8662567, -39.2489357, 38.6557350
2: -8.9014492, 30.2400932, -9.1305141, 31.0644073, -39.9658546, 39.3706055
3: -13.1529007, 30.8414726, -13.5239582, 31.7180500, -44.8709488, 44.3654327
4: -13.6967449, 30.2116623, -14.0812511, 31.0096245, -44.7063675, 44.2929077

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5901741, upper bound: 43.5296437
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.1871862, 26.6099091, -7.9041195, 29.0763073, -36.2634926, 34.5140305
1: -8.3826809, 30.0424767, -9.2882051, 32.8371925, -41.2198677, 39.3306808
2: -8.9014492, 30.2400932, -9.7911968, 33.1104507, -42.0119019, 40.0312881
3: -13.1529007, 30.8414726, -14.5323257, 33.7886276, -46.9415283, 45.3737984
4: -13.6967449, 30.2116623, -15.0085173, 33.1586647, -46.8554077, 45.2201729

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5901741, upper bound: 43.5296437
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.7295799, 28.3989639, -7.3701210, 27.3269844, -35.0565643, 35.7690849
1: -9.0717134, 32.0508652, -8.6132603, 30.8662567, -39.9379578, 40.6641235
2: -9.5743036, 32.3409538, -9.1305141, 31.0644073, -40.6387062, 41.4714661
3: -14.1834040, 32.9562950, -13.5239582, 31.7180500, -45.9014473, 46.4802551
4: -14.6434193, 32.4114075, -14.0812511, 31.0096245, -45.6530418, 46.4926529

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.7295799, 28.3989639, -7.9041195, 29.0763073, -36.8058853, 36.3030815
1: -9.0717134, 32.0508652, -9.2882051, 32.8371925, -41.9088936, 41.3390694
2: -9.5743036, 32.3409538, -9.7911968, 33.1104507, -42.6847534, 42.1321487
3: -14.1834040, 32.9562950, -14.5323257, 33.7886276, -47.9720230, 47.4886169
4: -14.6434193, 32.4114075, -15.0085173, 33.1586647, -47.8020859, 47.4199181

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.3701210, 27.3269844, -7.1871862, 26.6099091, -33.9800262, 34.5141678
1: -8.6132603, 30.8662567, -8.3826809, 30.0424767, -38.6557350, 39.2489357
2: -9.1305141, 31.0644073, -8.9014492, 30.2400932, -39.3706055, 39.9658546
3: -13.5239582, 31.7180500, -13.1529007, 30.8414726, -44.3654327, 44.8709488
4: -14.0812511, 31.0096245, -13.6967449, 30.2116623, -44.2929077, 44.7063675

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.3701210, 27.3269844, -7.7295799, 28.3989639, -35.7690849, 35.0565643
1: -8.6132603, 30.8662567, -9.0717134, 32.0508652, -40.6641235, 39.9379578
2: -9.1305141, 31.0644073, -9.5743036, 32.3409538, -41.4714661, 40.6387062
3: -13.5239582, 31.7180500, -14.1834040, 32.9562950, -46.4802551, 45.9014473
4: -14.0812511, 31.0096245, -14.6434193, 32.4114075, -46.4926529, 45.6530418

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.9041195, 29.0763073, -7.1871862, 26.6099091, -34.5140305, 36.2634926
1: -9.2882051, 32.8371925, -8.3826809, 30.0424767, -39.3306808, 41.2198677
2: -9.7911968, 33.1104507, -8.9014492, 30.2400932, -40.0312881, 42.0119019
3: -14.5323257, 33.7886276, -13.1529007, 30.8414726, -45.3737984, 46.9415283
4: -15.0085173, 33.1586647, -13.6967449, 30.2116623, -45.2201729, 46.8554077

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.42 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.9041195, 29.0763073, -7.7295799, 28.3989639, -36.3030853, 36.8058853
1: -9.2882051, 32.8371925, -9.0717134, 32.0508652, -41.3390694, 41.9088936
2: -9.7911968, 33.1104507, -9.5743036, 32.3409538, -42.1321487, 42.6847534
3: -14.5323257, 33.7886276, -14.1834040, 32.9562950, -47.4886169, 47.9720230
4: -15.0085173, 33.1586647, -14.6434193, 32.4114075, -47.4199181, 47.8020859

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3701210, 27.3269844, -7.3701210, 27.3269844, -34.6971054, 34.6971054
1: -8.6132603, 30.8662567, -8.6132603, 30.8662567, -39.4795113, 39.4795113
2: -9.1305141, 31.0644073, -9.1305141, 31.0644073, -40.1949158, 40.1949196
3: -13.5239582, 31.7180500, -13.5239582, 31.7180500, -45.2420082, 45.2420082
4: -14.0812511, 31.0096245, -14.0812511, 31.0096245, -45.0908699, 45.0908699

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5224054
time: 0.44 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3701210, 27.3269844, -7.9041195, 29.0763073, -36.4464264, 35.2311020
1: -8.6132603, 30.8662567, -9.2882051, 32.8371925, -41.4504471, 40.1544571
2: -9.1305141, 31.0644073, -9.7911968, 33.1104507, -42.2409668, 40.8556061
3: -13.5239582, 31.7180500, -14.5323257, 33.7886276, -47.3125839, 46.2503738
4: -14.0812511, 31.0096245, -15.0085173, 33.1586647, -47.2399139, 46.0181351

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5224054
time: 0.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.43 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.9041195, 29.0763073, -7.3701210, 27.3269844, -35.2311020, 36.4464264
1: -9.2882051, 32.8371925, -8.6132603, 30.8662567, -40.1544571, 41.4504471
2: -9.7911968, 33.1104507, -9.1305141, 31.0644073, -40.8556061, 42.2409668
3: -14.5323257, 33.7886276, -13.5239582, 31.7180500, -46.2503738, 47.3125839
4: -15.0085173, 33.1586647, -14.0812511, 31.0096245, -46.0181351, 47.2399139

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.9041195, 29.0763073, -7.9041195, 29.0763073, -36.9804268, 36.9804268
1: -9.2882051, 32.8371925, -9.2882051, 32.8371925, -42.1253929, 42.1253929
2: -9.7911968, 33.1104507, -9.7911968, 33.1104507, -42.9016495, 42.9016495
3: -14.5323257, 33.7886276, -14.5323257, 33.7886276, -48.3209496, 48.3209496
4: -15.0085173, 33.1586647, -15.0085173, 33.1586647, -48.1671791, 48.1671791

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.83 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5901741, upper bound: 43.5296437
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5901741, upper bound: 43.5296437
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5545765, upper bound: 43.5200645
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5224054
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5224054
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.1871862, 26.6099091, -33.3433685, 32.2624207
1: -7.8301010, 28.3211842, -8.3826809, 30.0424767, -37.8725777, 36.7038651
2: -8.3398962, 28.4661274, -8.9014492, 30.2400932, -38.5799904, 37.3675766
3: -12.3204012, 29.0548534, -13.1529007, 30.8414726, -43.1618729, 42.2077560
4: -12.9191322, 28.3510303, -13.6967449, 30.2116623, -43.1307907, 42.0477753

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -7.1871862, 26.6099091, -33.9370155, 34.4277458
1: -8.5321484, 30.8015041, -8.3826809, 30.0424767, -38.5746231, 39.1841850
2: -9.0758047, 30.9408817, -8.9014492, 30.2400932, -39.3158951, 39.8423309
3: -13.4046860, 31.6406326, -13.1529007, 30.8414726, -44.2461586, 44.7935295
4: -14.0712585, 30.7746468, -13.6967449, 30.2116623, -44.2829056, 44.4713898

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.7295799, 28.3989639, -35.1324234, 32.8048172
1: -7.8301010, 28.3211842, -9.0717134, 32.0508652, -39.8809662, 37.3928909
2: -8.3398962, 28.4661274, -9.5743036, 32.3409538, -40.6808510, 38.0404320
3: -12.3204012, 29.0548534, -14.1834040, 32.9562950, -45.2766953, 43.2382507
4: -12.9191322, 28.3510303, -14.6434193, 32.4114075, -45.3305359, 42.9944496

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -7.7295799, 28.3989639, -35.7260742, 34.9701462
1: -8.5321484, 30.8015041, -9.0717134, 32.0508652, -40.5830154, 39.8732147
2: -9.0758047, 30.9408817, -9.5743036, 32.3409538, -41.4167557, 40.5151863
3: -13.4046860, 31.6406326, -14.1834040, 32.9562950, -46.3609810, 45.8240280
4: -14.0712585, 30.7746468, -14.6434193, 32.4114075, -46.4826508, 45.4180641

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5691259, upper bound: 43.5588638
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -7.1871862, 26.6099091, -33.1162109, 31.4548264
1: -7.5842962, 27.4015598, -8.3826809, 30.0424767, -37.6267662, 35.7842407
2: -8.0552320, 27.5818462, -8.9014492, 30.2400932, -38.2953262, 36.4832954
3: -11.9514847, 28.1189060, -13.1529007, 30.8414726, -42.7929573, 41.2718048
4: -12.5255804, 27.4433079, -13.6967449, 30.2116623, -42.7372398, 41.1400528

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -7.1871862, 26.6099091, -35.5347290, 40.3725357
1: -10.5382071, 37.6687279, -8.3826809, 30.0424767, -40.5806808, 46.0514069
2: -11.0602732, 37.7767830, -8.9014492, 30.2400932, -41.3003654, 46.6782303
3: -16.4940701, 38.9501419, -13.1529007, 30.8414726, -47.3355408, 52.1030426
4: -17.3545589, 37.3720055, -13.6967449, 30.2116623, -47.5662193, 51.0687485

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -7.7295799, 28.3989639, -34.9052658, 31.9972191
1: -7.5842962, 27.4015598, -9.0717134, 32.0508652, -39.6351585, 36.4732628
2: -8.0552320, 27.5818462, -9.5743036, 32.3409538, -40.3961868, 37.1561470
3: -11.9514847, 28.1189060, -14.1834040, 32.9562950, -44.9077797, 42.3023033
4: -12.5255804, 27.4433079, -14.6434193, 32.4114075, -44.9369850, 42.0867271

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -7.7295799, 28.3989639, -37.3237801, 40.9149323
1: -10.5382071, 37.6687279, -9.0717134, 32.0508652, -42.5890732, 46.7404404
2: -11.0602732, 37.7767830, -9.5743036, 32.3409538, -43.4012260, 47.3510857
3: -16.4940701, 38.9501419, -14.1834040, 32.9562950, -49.4503632, 53.1335411
4: -17.3545589, 37.3720055, -14.6434193, 32.4114075, -49.7659645, 52.0154228

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.3701210, 27.3269844, -34.0604477, 32.4453583
1: -7.8301010, 28.3211842, -8.6132603, 30.8662567, -38.6963577, 36.9344444
2: -8.3398962, 28.4661274, -9.1305141, 31.0644073, -39.4043045, 37.5966415
3: -12.3204012, 29.0548534, -13.5239582, 31.7180500, -44.0384521, 42.5788116
4: -12.9191322, 28.3510303, -14.0812511, 31.0096245, -43.9287529, 42.4322815

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -7.3701210, 27.3269844, -34.6540985, 34.6106834
1: -8.5321484, 30.8015041, -8.6132603, 30.8662567, -39.3984032, 39.4147644
2: -9.0758047, 30.9408817, -9.1305141, 31.0644073, -40.1402054, 40.0713959
3: -13.4046860, 31.6406326, -13.5239582, 31.7180500, -45.1227341, 45.1645889
4: -14.0712585, 30.7746468, -14.0812511, 31.0096245, -45.0808716, 44.8558922

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5672675, upper bound: 43.5546301
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.9041195, 29.0763073, -35.8097687, 32.9793549
1: -7.8301010, 28.3211842, -9.2882051, 32.8371925, -40.6672935, 37.6093903
2: -8.3398962, 28.4661274, -9.7911968, 33.1104507, -41.4503479, 38.2573242
3: -12.3204012, 29.0548534, -14.5323257, 33.7886276, -46.1090279, 43.5871773
4: -12.9191322, 28.3510303, -15.0085173, 33.1586647, -46.0777969, 43.3595467

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -7.9041195, 29.0763073, -36.4034271, 35.1446838
1: -8.5321484, 30.8015041, -9.2882051, 32.8371925, -41.3693390, 40.0897102
2: -9.0758047, 30.9408817, -9.7911968, 33.1104507, -42.1862564, 40.7320786
3: -13.4046860, 31.6406326, -14.5323257, 33.7886276, -47.1933136, 46.1729546
4: -14.0712585, 30.7746468, -15.0085173, 33.1586647, -47.2299118, 45.7831573

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5654680, upper bound: 43.5237682
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -7.3701210, 27.3269844, -33.8332863, 31.6377602
1: -7.5842962, 27.4015598, -8.6132603, 30.8662567, -38.4505424, 36.0148125
2: -8.0552320, 27.5818462, -9.1305141, 31.0644073, -39.1196404, 36.7123566
3: -11.9514847, 28.1189060, -13.5239582, 31.7180500, -43.6695328, 41.6428642
4: -12.5255804, 27.4433079, -14.0812511, 31.0096245, -43.5352020, 41.5245514

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -7.3701210, 27.3269844, -36.2518044, 40.5554695
1: -10.5382071, 37.6687279, -8.6132603, 30.8662567, -41.4044571, 46.2819901
2: -11.0602732, 37.7767830, -9.1305141, 31.0644073, -42.1246796, 46.9072952
3: -16.4940701, 38.9501419, -13.5239582, 31.7180500, -48.2121201, 52.4740982
4: -17.3545589, 37.3720055, -14.0812511, 31.0096245, -48.3641815, 51.4532509

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -7.9041195, 29.0763073, -35.5826111, 32.1717606
1: -7.5842962, 27.4015598, -9.2882051, 32.8371925, -40.4214783, 36.6897659
2: -8.0552320, 27.5818462, -9.7911968, 33.1104507, -41.1656837, 37.3730431
3: -11.9514847, 28.1189060, -14.5323257, 33.7886276, -45.7401123, 42.6512299
4: -12.5255804, 27.4433079, -15.0085173, 33.1586647, -45.6842461, 42.4518204

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -7.9041195, 29.0763073, -38.0011292, 41.0894699
1: -10.5382071, 37.6687279, -9.2882051, 32.8371925, -43.3753929, 46.9569321
2: -11.0602732, 37.7767830, -9.7911968, 33.1104507, -44.1707230, 47.5679779
3: -16.4940701, 38.9501419, -14.5323257, 33.7886276, -50.2826996, 53.4824677
4: -17.3545589, 37.3720055, -15.0085173, 33.1586647, -50.5132217, 52.3805161

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -7.1871862, 26.6099091, -32.7415123, 30.3283634
1: -7.1131840, 26.1596413, -8.3826809, 30.0424767, -37.1556625, 34.5423203
2: -7.5950131, 26.2545033, -8.9014492, 30.2400932, -37.8351059, 35.1559525
3: -11.2687273, 26.8291836, -13.1529007, 30.8414726, -42.1101990, 39.9820786
4: -11.9468098, 25.9775772, -13.6967449, 30.2116623, -42.1584702, 39.6743240

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -7.1871862, 26.6099091, -35.1427193, 39.2071724
1: -10.0470028, 36.3811836, -8.3826809, 30.0424767, -40.0894775, 44.7638626
2: -10.5810204, 36.3961143, -8.9014492, 30.2400932, -40.8211136, 45.2975616
3: -15.7773991, 37.6067963, -13.1529007, 30.8414726, -46.6188736, 50.7596970
4: -16.7466125, 35.8476906, -13.6967449, 30.2116623, -46.9582710, 49.5444336

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -7.7295799, 28.3989639, -34.5305672, 30.8707561
1: -7.1131840, 26.1596413, -9.0717134, 32.0508652, -39.1640472, 35.2313538
2: -7.5950131, 26.2545033, -9.5743036, 32.3409538, -39.9359665, 35.8288040
3: -11.2687273, 26.8291836, -14.1834040, 32.9562950, -44.2250214, 41.0125771
4: -11.9468098, 25.9775772, -14.6434193, 32.4114075, -44.3582153, 40.6209946

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -7.7295799, 28.3989639, -36.9317780, 39.7495651
1: -10.0470028, 36.3811836, -9.0717134, 32.0508652, -42.0978699, 45.4528923
2: -10.5810204, 36.3961143, -9.5743036, 32.3409538, -42.9219742, 45.9704170
3: -15.7773991, 37.6067963, -14.1834040, 32.9562950, -48.7336960, 51.7901993
4: -16.7466125, 35.8476906, -14.6434193, 32.4114075, -49.1580162, 50.4911118

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.42 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -7.1871862, 26.6099091, -33.2752991, 32.0736694
1: -7.7828097, 28.1194229, -8.3826809, 30.0424767, -37.8252754, 36.5021057
2: -8.2539454, 28.2906685, -8.9014492, 30.2400932, -38.4940376, 37.1921158
3: -12.2706099, 28.8829918, -13.1529007, 30.8414726, -43.1120834, 42.0358925
4: -12.8631258, 28.1254616, -13.6967449, 30.2116623, -43.0747833, 41.8222046

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -7.1871862, 26.6099091, -35.7214928, 41.0997276
1: -10.7727814, 38.5158424, -8.3826809, 30.0424767, -40.8152580, 46.8985214
2: -11.2928276, 38.6079216, -8.9014492, 30.2400932, -41.5329208, 47.5093689
3: -16.8676147, 39.8401985, -13.1529007, 30.8414726, -47.7090874, 52.9930992
4: -17.7444897, 38.1677361, -13.6967449, 30.2116623, -47.9561501, 51.8644791

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -7.7295799, 28.3989639, -35.0643578, 32.6160660
1: -7.7828097, 28.1194229, -9.0717134, 32.0508652, -39.8336678, 37.1911354
2: -8.2539454, 28.2906685, -9.5743036, 32.3409538, -40.5948982, 37.8649712
3: -12.2706099, 28.8829918, -14.1834040, 32.9562950, -45.2269058, 43.0663910
4: -12.8631258, 28.1254616, -14.6434193, 32.4114075, -45.2745285, 42.7688828

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -7.7295799, 28.3989639, -37.5105515, 41.6421204
1: -10.7727814, 38.5158424, -9.0717134, 32.0508652, -42.8236465, 47.5875511
2: -11.2928276, 38.6079216, -9.5743036, 32.3409538, -43.6337814, 48.1822243
3: -16.8676147, 39.8401985, -14.1834040, 32.9562950, -49.8239098, 54.0235939
4: -17.7444897, 38.1677361, -14.6434193, 32.4114075, -50.1558952, 52.8111572

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -7.3701210, 27.3269844, -33.4585876, 30.5112972
1: -7.1131840, 26.1596413, -8.6132603, 30.8662567, -37.9794388, 34.7728996
2: -7.5950131, 26.2545033, -9.1305141, 31.0644073, -38.6594200, 35.3850136
3: -11.2687273, 26.8291836, -13.5239582, 31.7180500, -42.9867783, 40.3531380
4: -11.9468098, 25.9775772, -14.0812511, 31.0096245, -42.9564323, 40.0588188

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -7.3701210, 27.3269844, -35.8597984, 39.3901062
1: -10.0470028, 36.3811836, -8.6132603, 30.8662567, -40.9132614, 44.9944458
2: -10.5810204, 36.3961143, -9.1305141, 31.0644073, -41.6454239, 45.5266266
3: -15.7773991, 37.6067963, -13.5239582, 31.7180500, -47.4954491, 51.1307526
4: -16.7466125, 35.8476906, -14.0812511, 31.0096245, -47.7562370, 49.9289398

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -7.9041195, 29.0763073, -35.2079163, 31.0452976
1: -7.1131840, 26.1596413, -9.2882051, 32.8371925, -39.9503708, 35.4478455
2: -7.5950131, 26.2545033, -9.7911968, 33.1104507, -40.7054634, 36.0457001
3: -11.2687273, 26.8291836, -14.5323257, 33.7886276, -45.0573540, 41.3615036
4: -11.9468098, 25.9775772, -15.0085173, 33.1586647, -45.1054764, 40.9860878

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -7.9041195, 29.0763073, -37.6091232, 39.9241028
1: -10.0470028, 36.3811836, -9.2882051, 32.8371925, -42.8841934, 45.6693878
2: -10.5810204, 36.3961143, -9.7911968, 33.1104507, -43.6914711, 46.1873093
3: -15.7773991, 37.6067963, -14.5323257, 33.7886276, -49.5660248, 52.1391220
4: -16.7466125, 35.8476906, -15.0085173, 33.1586647, -49.9052773, 50.8562050

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -7.3701210, 27.3269844, -33.9923782, 32.2566032
1: -7.7828097, 28.1194229, -8.6132603, 30.8662567, -38.6490517, 36.7326813
2: -8.2539454, 28.2906685, -9.1305141, 31.0644073, -39.3183479, 37.4211807
3: -12.2706099, 28.8829918, -13.5239582, 31.7180500, -43.9886589, 42.4069519
4: -12.8631258, 28.1254616, -14.0812511, 31.0096245, -43.8727493, 42.2067108

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -7.3701210, 27.3269844, -36.4385719, 41.2826614
1: -10.7727814, 38.5158424, -8.6132603, 30.8662567, -41.6390381, 47.1291008
2: -11.2928276, 38.6079216, -9.1305141, 31.0644073, -42.3572350, 47.7384338
3: -16.8676147, 39.8401985, -13.5239582, 31.7180500, -48.5856628, 53.3641586
4: -17.7444897, 38.1677361, -14.0812511, 31.0096245, -48.7541122, 52.2489853

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -7.9041195, 29.0763073, -35.7416992, 32.7906036
1: -7.7828097, 28.1194229, -9.2882051, 32.8371925, -40.6199875, 37.4076271
2: -8.2539454, 28.2906685, -9.7911968, 33.1104507, -41.3643951, 38.0818634
3: -12.2706099, 28.8829918, -14.5323257, 33.7886276, -46.0592384, 43.4153175
4: -12.8631258, 28.1254616, -15.0085173, 33.1586647, -46.0217896, 43.1339760

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -7.9041195, 29.0763073, -38.1878967, 41.8166580
1: -10.7727814, 38.5158424, -9.2882051, 32.8371925, -43.6099701, 47.8040466
2: -11.2928276, 38.6079216, -9.7911968, 33.1104507, -44.4032784, 48.3991165
3: -16.8676147, 39.8401985, -14.5323257, 33.7886276, -50.6562424, 54.3725204
4: -17.7444897, 38.1677361, -15.0085173, 33.1586647, -50.9031525, 53.1762505

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.11 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5790101, upper bound: 43.5790101
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5778470, upper bound: 43.5612017
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5691259, upper bound: 43.5588638
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5569309, upper bound: 43.5676053
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5759886, upper bound: 43.5569681
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5672675, upper bound: 43.5546301
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5535632, upper bound: 43.5437893
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5441351, upper bound: 43.5645834
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5437893, upper bound: 43.5535632
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5188589, upper bound: 43.5625398
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5185132, upper bound: 43.5515196
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5407674, upper bound: 43.5407674
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5154913, upper bound: 43.5387238
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -6.7334614, 25.0752354, -31.8086967, 31.8086967
1: -7.8301010, 28.3211842, -7.8301010, 28.3211842, -36.1512833, 36.1512833
2: -8.3398962, 28.4661274, -8.3398962, 28.4661274, -36.8060226, 36.8060226
3: -12.3204012, 29.0548534, -12.3204012, 29.0548534, -41.3752556, 41.3752556
4: -12.9191322, 28.3510303, -12.9191322, 28.3510303, -41.2701645, 41.2701645

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5949951, upper bound: 43.5825477
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5819614, upper bound: 43.5792965
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.3271179, 27.2405643, -33.9740257, 32.4023514
1: -7.8301010, 28.3211842, -8.5321484, 30.8015041, -38.6316071, 36.8533325
2: -8.3398962, 28.4661274, -9.0758047, 30.9408817, -39.2807770, 37.5419312
3: -12.3204012, 29.0548534, -13.4046860, 31.6406326, -43.9610329, 42.4595413
4: -12.9191322, 28.3510303, -14.0712585, 30.7746468, -43.6937752, 42.4222794

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5949951, upper bound: 43.5825477
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5819614, upper bound: 43.5792965
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -6.7334614, 25.0752354, -32.4023514, 33.9740257
1: -8.5321484, 30.8015041, -7.8301010, 28.3211842, -36.8533325, 38.6316071
2: -9.0758047, 30.9408817, -8.3398962, 28.4661274, -37.5419312, 39.2807770
3: -13.4046860, 31.6406326, -12.3204012, 29.0548534, -42.4595413, 43.9610329
4: -14.0712585, 30.7746468, -12.9191322, 28.3510303, -42.4222794, 43.6937752

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5744021, upper bound: 43.5279582
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5233502, upper bound: 43.5233502
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -7.3271179, 27.2405643, -34.5676727, 34.5676727
1: -8.5321484, 30.8015041, -8.5321484, 30.8015041, -39.3336525, 39.3336525
2: -9.0758047, 30.9408817, -9.0758047, 30.9408817, -40.0166855, 40.0166855
3: -13.4046860, 31.6406326, -13.4046860, 31.6406326, -45.0453186, 45.0453186
4: -14.0712585, 30.7746468, -14.0712585, 30.7746468, -44.8458900, 44.8458900

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5744021, upper bound: 43.5286222
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5251243, upper bound: 43.5241632
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.2693892, 26.8420372, -33.5755005, 32.3446236
1: -7.8301010, 28.3211842, -8.5089540, 30.3026695, -38.1327705, 36.8301392
2: -8.3398962, 28.4661274, -9.0044117, 30.5334988, -38.8733940, 37.4705391
3: -12.3204012, 29.0548534, -13.3364210, 31.1392040, -43.4596024, 42.3912735
4: -12.9191322, 28.3510303, -13.8503227, 30.5232334, -43.4423637, 42.2013550

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5807983, upper bound: 43.5614882
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.8313341, 28.8915958, -35.6250572, 32.9065704
1: -7.8301010, 28.3211842, -9.1737223, 32.6585579, -40.4886589, 37.4949036
2: -8.3398962, 28.4661274, -9.7022839, 32.8745689, -41.2144661, 38.1684113
3: -12.3204012, 29.0548534, -14.3621759, 33.5927696, -45.9131699, 43.4170265
4: -12.9191322, 28.3510303, -14.9443817, 32.8122597, -45.7313843, 43.2954102

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5938320, upper bound: 43.5647393
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5807983, upper bound: 43.5614882
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -6.5063033, 24.2676392, -31.5947571, 33.7468681
1: -8.5321484, 30.8015041, -7.5842962, 27.4015598, -35.9337082, 38.3857994
2: -9.0758047, 30.9408817, -8.0552320, 27.5818462, -36.6576462, 38.9961128
3: -13.4046860, 31.6406326, -11.9514847, 28.1189060, -41.5235901, 43.5921173
4: -14.0712585, 30.7746468, -12.5255804, 27.4433079, -41.5145531, 43.3002243

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -8.9248209, 33.1853523, -40.5124702, 36.1653824
1: -8.5321484, 30.8015041, -10.5382071, 37.6687279, -46.2008743, 41.3397102
2: -9.0758047, 30.9408817, -11.0602732, 37.7767830, -46.8525887, 42.0011559
3: -13.4046860, 31.6406326, -16.4940701, 38.9501419, -52.3548279, 48.1347008
4: -14.0712585, 30.7746468, -17.3545589, 37.3720055, -51.4432487, 48.1292038

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5676053, upper bound: 43.5567416
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -5.9647384, 22.4902878, -28.9965916, 30.2323761
1: -7.5842962, 27.4015598, -6.9048386, 25.4210529, -33.0053444, 34.3063965
2: -8.0552320, 27.5818462, -7.3848820, 25.5047226, -33.5599556, 34.9667244
3: -11.9514847, 28.1189060, -10.9337454, 26.0330429, -37.9845276, 39.0526505
4: -12.5255804, 27.4433079, -11.5946884, 25.2559605, -37.7815361, 39.0379829

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5599878, upper bound: 43.5691566
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5599878, upper bound: 43.5691566
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -8.3492012, 31.2999210, -37.8062248, 32.6168404
1: -7.5842962, 27.4015598, -9.8115788, 35.5474586, -43.1317520, 37.2131348
2: -8.0552320, 27.5818462, -10.3486872, 35.5690842, -43.6243172, 37.9305344
3: -11.9514847, 28.1189060, -15.4028683, 36.7234039, -48.6748848, 43.5217743
4: -12.5255804, 27.4433079, -16.3597260, 35.0432816, -47.5688629, 43.8030281

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5599878, upper bound: 43.5691566
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5599878, upper bound: 43.5691566
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -5.9647384, 22.4902878, -31.4151058, 39.1500854
1: -10.5382071, 37.6687279, -6.9048386, 25.4210529, -35.9592590, 44.5735664
2: -11.0602732, 37.7767830, -7.3848820, 25.5047226, -36.5649910, 45.1616669
3: -16.4940701, 38.9501419, -10.9337454, 26.0330429, -42.5271149, 49.8838844
4: -17.3545589, 37.3720055, -11.5946884, 25.2559605, -42.6105156, 48.9666862

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5265996
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5228568
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -8.3492012, 31.2999210, -40.2247429, 41.5345459
1: -10.5382071, 37.6687279, -9.8115788, 35.5474586, -46.0856628, 47.4803085
2: -11.0602732, 37.7767830, -10.3486872, 35.5690842, -46.6293564, 48.1254692
3: -16.4940701, 38.9501419, -15.4028683, 36.7234039, -53.2174759, 54.3530083
4: -17.3545589, 37.3720055, -16.3597260, 35.0432816, -52.3978424, 53.7317276

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5265996
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5228568
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -6.5063033, 24.2676392, -30.7739429, 30.7739429
1: -7.5842962, 27.4015598, -7.5842962, 27.4015598, -34.9858475, 34.9858475
2: -8.0552320, 27.5818462, -8.0552320, 27.5818462, -35.6370773, 35.6370773
3: -11.9514847, 28.1189060, -11.9514847, 28.1189060, -40.0703888, 40.0703888
4: -12.5255804, 27.4433079, -12.5255804, 27.4433079, -39.9688873, 39.9688873

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585932, upper bound: 43.5579472
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585180, upper bound: 43.5579165
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -8.9248209, 33.1853523, -39.6916504, 33.1924591
1: -7.5842962, 27.4015598, -10.5382071, 37.6687279, -45.2530251, 37.9397621
2: -8.0552320, 27.5818462, -11.0602732, 37.7767830, -45.8320160, 38.6421165
3: -11.9514847, 28.1189060, -16.4940701, 38.9501419, -50.9016266, 44.6129761
4: -12.5255804, 27.4433079, -17.3545589, 37.3720055, -49.8975830, 44.7978668

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585932, upper bound: 43.5579472
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585180, upper bound: 43.5579165
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -6.5063033, 24.2676392, -33.1924591, 39.6916504
1: -10.5382071, 37.6687279, -7.5842962, 27.4015598, -37.9397621, 45.2530251
2: -11.0602732, 37.7767830, -8.0552320, 27.5818462, -38.6421165, 45.8320160
3: -16.4940701, 38.9501419, -11.9514847, 28.1189060, -44.6129761, 50.9016266
4: -17.3545589, 37.3720055, -12.5255804, 27.4433079, -44.7978630, 49.8975830

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5260811
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5223382
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -8.9248209, 33.1853523, -42.1101685, 42.1101685
1: -10.5382071, 37.6687279, -10.5382071, 37.6687279, -48.2069359, 48.2069359
2: -11.0602732, 37.7767830, -11.0602732, 37.7767830, -48.8370552, 48.8370552
3: -16.4940701, 38.9501419, -16.4940701, 38.9501419, -55.4442139, 55.4442139
4: -17.3545589, 37.3720055, -17.3545589, 37.3720055, -54.7265625, 54.7265625

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5527447, upper bound: 43.5260811
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5223382
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -6.9100652, 25.7693195, -32.5027809, 31.9853001
1: -7.8301010, 28.3211842, -8.0518236, 29.1186275, -36.9487305, 36.3730049
2: -8.3398962, 28.4661274, -8.5616179, 29.2663651, -37.6062622, 37.0277443
3: -12.3204012, 29.0548534, -12.6775093, 29.9030113, -42.2234116, 41.7323608
4: -12.9191322, 28.3510303, -13.2914982, 29.1219196, -42.0410385, 41.6425247

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5919736, upper bound: 43.5605057
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5789399, upper bound: 43.5572545
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.5059209, 27.9394646, -34.6729279, 32.5811577
1: -7.8301010, 28.3211842, -8.7573853, 31.6072998, -39.4374008, 37.0785637
2: -8.3398962, 28.4661274, -9.2995558, 31.7472954, -40.0871925, 37.7656822
3: -12.3204012, 29.0548534, -13.7668066, 32.4964180, -44.8168182, 42.8216591
4: -12.9191322, 28.3510303, -14.4473610, 31.5514050, -44.4705353, 42.7983932

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5919736, upper bound: 43.5605057
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5789399, upper bound: 43.5572545
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -6.1316075, 23.1411781, -30.4682961, 33.3721657
1: -8.5321484, 30.8015041, -7.1131840, 26.1596413, -34.6917877, 37.9146881
2: -9.0758047, 30.9408817, -7.5950131, 26.2545033, -35.3303032, 38.5358963
3: -13.4046860, 31.6406326, -11.2687273, 26.8291836, -40.2338676, 42.9093590
4: -14.0712585, 30.7746468, -11.9468098, 25.9775772, -40.0488205, 42.7214546

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -8.5328178, 32.0199852, -39.3471031, 35.7733765
1: -8.5321484, 30.8015041, -10.0470028, 36.3811836, -44.9133301, 40.8485069
2: -9.0758047, 30.9408817, -10.5810204, 36.3961143, -45.4719124, 41.5219002
3: -13.4046860, 31.6406326, -15.7773991, 37.6067963, -51.0114822, 47.4180298
4: -14.0712585, 30.7746468, -16.7466125, 35.8476906, -49.9189377, 47.5212555

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645834, upper bound: 43.5441351
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -7.4370813, 27.4980087, -34.2314682, 32.5123177
1: -7.8301010, 28.3211842, -8.7175827, 31.0629864, -38.8930893, 37.0387650
2: -8.3398962, 28.4661274, -9.2134495, 31.2856255, -39.6255226, 37.6795769
3: -12.3204012, 29.0548534, -13.6725578, 31.9439163, -44.2643166, 42.7274055
4: -12.9191322, 28.3510303, -14.2024050, 31.2481651, -44.1672935, 42.5534363

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5901741, upper bound: 43.5296437
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5771404, upper bound: 43.5263926
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.7334614, 25.0752354, -8.0359106, 29.6763287, -36.4097900, 33.1111450
1: -7.8301010, 28.3211842, -9.4270086, 33.5660286, -41.3961296, 37.7481918
2: -8.3398962, 28.4661274, -9.9561176, 33.7784920, -42.1183891, 38.4222450
3: -12.3204012, 29.0548534, -14.7677183, 34.5513916, -46.8717918, 43.8225708
4: -12.9191322, 28.3510303, -15.3659382, 33.6877899, -46.6069221, 43.7169685

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5901741, upper bound: 43.5296437
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5771404, upper bound: 43.5263926
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -7.4370813, 27.4980087, -34.8251190, 34.6776428
1: -8.5321484, 30.8015041, -8.7175827, 31.0629864, -39.5951347, 39.5190887
2: -9.0758047, 30.9408817, -9.2134495, 31.2856255, -40.3614311, 40.1543312
3: -13.4046860, 31.6406326, -13.6725578, 31.9439163, -45.3486023, 45.3131866
4: -14.0712585, 30.7746468, -14.2024050, 31.2481651, -45.3194122, 44.9770508

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5231372, upper bound: 43.5214981
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.3271179, 27.2405643, -8.0359106, 29.6763287, -37.0034485, 35.2764664
1: -8.5321484, 30.8015041, -9.4270086, 33.5660286, -42.0981750, 40.2285118
2: -9.0758047, 30.9408817, -9.9561176, 33.7784920, -42.8542900, 40.8969955
3: -13.4046860, 31.6406326, -14.7677183, 34.5513916, -47.9560776, 46.4083481
4: -14.0712585, 30.7746468, -15.3659382, 33.6877899, -47.7590370, 46.1405792

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5741891, upper bound: 43.5261061
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5249113, upper bound: 43.5214981
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -6.1316075, 23.1411781, -29.6474819, 30.3992424
1: -7.5842962, 27.4015598, -7.1131840, 26.1596413, -33.7439346, 34.5147400
2: -8.0552320, 27.5818462, -7.5950131, 26.2545033, -34.3097343, 35.1768570
3: -11.9514847, 28.1189060, -11.2687273, 26.8291836, -38.7806625, 39.3876343
4: -12.5255804, 27.4433079, -11.9468098, 25.9775772, -38.5031548, 39.3901176

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555713, upper bound: 43.5453406
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5554961, upper bound: 43.5453099
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -8.5328178, 32.0199852, -38.5262871, 32.8004570
1: -7.5842962, 27.4015598, -10.0470028, 36.3811836, -43.9654770, 37.4485626
2: -8.0552320, 27.5818462, -10.5810204, 36.3961143, -44.4513474, 38.1628647
3: -11.9514847, 28.1189060, -15.7773991, 37.6067963, -49.5582809, 43.8963051
4: -12.5255804, 27.4433079, -16.7466125, 35.8476906, -48.3732719, 44.1899185

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555713, upper bound: 43.5453406
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5554961, upper bound: 43.5453099
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -6.1316075, 23.1411781, -32.0659981, 39.3169518
1: -10.5382071, 37.6687279, -7.1131840, 26.1596413, -36.6978455, 44.7819138
2: -11.0602732, 37.7767830, -7.5950131, 26.2545033, -37.3147774, 45.3717957
3: -16.4940701, 38.9501419, -11.2687273, 26.8291836, -43.3232498, 50.2188683
4: -17.3545589, 37.3720055, -11.9468098, 25.9775772, -43.3321342, 49.3188133

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5523180, upper bound: 43.5263945
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218139, upper bound: 43.5226517
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -8.5328178, 32.0199852, -40.9448051, 41.7181664
1: -10.5382071, 37.6687279, -10.0470028, 36.3811836, -46.9193916, 47.7157288
2: -11.0602732, 37.7767830, -10.5810204, 36.3961143, -47.4563866, 48.3578033
3: -16.4940701, 38.9501419, -15.7773991, 37.6067963, -54.1008682, 54.7275391
4: -17.3545589, 37.3720055, -16.7466125, 35.8476906, -53.2022476, 54.1186142

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5523180, upper bound: 43.5263945
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218139, upper bound: 43.5226517
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -6.6653934, 24.8864861, -31.3927898, 30.9330311
1: -7.5842962, 27.4015598, -7.7828097, 28.1194229, -35.7037201, 35.1843605
2: -8.0552320, 27.5818462, -8.2539454, 28.2906685, -36.3459015, 35.8357849
3: -11.9514847, 28.1189060, -12.2706099, 28.8829918, -40.8344765, 40.3895149
4: -12.5255804, 27.4433079, -12.8631258, 28.1254616, -40.6510429, 40.3064346

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535277, upper bound: 43.5200645
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5534525, upper bound: 43.5200338
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.5063033, 24.2676392, -9.1115875, 33.9125404, -40.4188423, 33.3792267
1: -7.5842962, 27.4015598, -10.7727814, 38.5158424, -46.1001358, 38.1743393
2: -8.0552320, 27.5818462, -11.2928276, 38.6079216, -46.6631546, 38.8746719
3: -11.9514847, 28.1189060, -16.8676147, 39.8401985, -51.7916832, 44.9865189
4: -12.5255804, 27.4433079, -17.7444897, 38.1677361, -50.6933174, 45.1877975

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5535277, upper bound: 43.5200645
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5534525, upper bound: 43.5200338
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -6.6653934, 24.8864861, -33.8113060, 39.8507423
1: -10.5382071, 37.6687279, -7.7828097, 28.1194229, -38.6576309, 45.4515343
2: -11.0602732, 37.7767830, -8.2539454, 28.2906685, -39.3509407, 46.0307274
3: -16.4940701, 38.9501419, -12.2706099, 28.8829918, -45.3770599, 51.2207527
4: -17.3545589, 37.3720055, -12.8631258, 28.1254616, -45.4800186, 50.2351265

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210155, upper bound: 43.5147703
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.9248209, 33.1853523, -9.1115875, 33.9125404, -42.8373604, 42.2969398
1: -10.5382071, 37.6687279, -10.7727814, 38.5158424, -49.0540466, 48.4415092
2: -11.0602732, 37.7767830, -11.2928276, 38.6079216, -49.6681938, 49.0696106
3: -16.4940701, 38.9501419, -16.8676147, 39.8401985, -56.3342667, 55.8177567
4: -17.3545589, 37.3720055, -17.7444897, 38.1677361, -55.5222931, 55.1164932

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5515196, upper bound: 43.5185132
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210155, upper bound: 43.5147703
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -5.9647384, 22.4902878, -28.6218910, 29.1059151
1: -7.1131840, 26.1596413, -6.9048386, 25.4210529, -32.5342369, 33.0644798
2: -7.5950131, 26.2545033, -7.3848820, 25.5047226, -33.0997314, 33.6393814
3: -11.2687273, 26.8291836, -10.9337454, 26.0330429, -37.3017693, 37.7629204
4: -11.9468098, 25.9775772, -11.5946884, 25.2559605, -37.2027702, 37.5722542

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5724913, upper bound: 43.5714975
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585753, upper bound: 43.5680569
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -8.3492012, 31.2999210, -37.4315300, 31.4903793
1: -7.1131840, 26.1596413, -9.8115788, 35.5474586, -42.6606445, 35.9712219
2: -7.5950131, 26.2545033, -10.3486872, 35.5690842, -43.1640968, 36.6031914
3: -11.2687273, 26.8291836, -15.4028683, 36.7234039, -47.9921303, 42.2320442
4: -11.9468098, 25.9775772, -16.3597260, 35.0432816, -46.9900894, 42.3372993

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5724913, upper bound: 43.5714975
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5585753, upper bound: 43.5680569
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -5.9647384, 22.4902878, -31.0231018, 37.9847221
1: -10.0470028, 36.3811836, -6.9048386, 25.4210529, -35.4680557, 43.2860222
2: -10.5810204, 36.3961143, -7.3848820, 25.5047226, -36.0857315, 43.7809944
3: -15.7773991, 37.6067963, -10.9337454, 26.0330429, -41.8104401, 48.5405426
4: -16.7466125, 35.8476906, -11.5946884, 25.2559605, -42.0025711, 47.4423752

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5235777
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5223325
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -8.3492012, 31.2999210, -39.8327370, 40.3691864
1: -10.0470028, 36.3811836, -9.8115788, 35.5474586, -45.5944595, 46.1927643
2: -10.5810204, 36.3961143, -10.3486872, 35.5690842, -46.1501045, 46.7448006
3: -15.7773991, 37.6067963, -15.4028683, 36.7234039, -52.5008011, 53.0096664
4: -16.7466125, 35.8476906, -16.3597260, 35.0432816, -51.7898941, 52.2074165

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5235777
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225541, upper bound: 43.5223325
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -6.5063033, 24.2676392, -30.3992424, 29.6474819
1: -7.1131840, 26.1596413, -7.5842962, 27.4015598, -34.5147400, 33.7439308
2: -7.5950131, 26.2545033, -8.0552320, 27.5818462, -35.1768608, 34.3097343
3: -11.2687273, 26.8291836, -11.9514847, 28.1189060, -39.3876343, 38.7806625
4: -11.9468098, 25.9775772, -12.5255804, 27.4433079, -39.3901138, 38.5031509

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5720115, upper bound: 43.5602881
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542844, upper bound: 43.5560581
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -8.9248209, 33.1853523, -39.3169556, 32.0659981
1: -7.1131840, 26.1596413, -10.5382071, 37.6687279, -44.7819138, 36.6978455
2: -7.5950131, 26.2545033, -11.0602732, 37.7767830, -45.3717957, 37.3147774
3: -11.2687273, 26.8291836, -16.4940701, 38.9501419, -50.2188683, 43.3232498
4: -11.9468098, 25.9775772, -17.3545589, 37.3720055, -49.3188133, 43.3321304

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5720115, upper bound: 43.5602881
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542844, upper bound: 43.5560581
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -6.5063033, 24.2676392, -32.8004570, 38.5262871
1: -10.0470028, 36.3811836, -7.5842962, 27.4015598, -37.4485626, 43.9654808
2: -10.5810204, 36.3961143, -8.0552320, 27.5818462, -38.1628647, 44.4513474
3: -15.7773991, 37.6067963, -11.9514847, 28.1189060, -43.8963051, 49.5582809
4: -16.7466125, 35.8476906, -12.5255804, 27.4433079, -44.1899185, 48.3732719

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5400465, upper bound: 43.5230592
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5226517, upper bound: 43.5218139
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -8.9248209, 33.1853523, -41.7181664, 40.9448051
1: -10.0470028, 36.3811836, -10.5382071, 37.6687279, -47.7157288, 46.9193916
2: -10.5810204, 36.3961143, -11.0602732, 37.7767830, -48.3578033, 47.4563866
3: -15.7773991, 37.6067963, -16.4940701, 38.9501419, -54.7275391, 54.1008682
4: -16.7466125, 35.8476906, -17.3545589, 37.3720055, -54.1186142, 53.2022476

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399489, upper bound: 43.5230592
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5226517, upper bound: 43.5218139
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -5.9647384, 22.4902878, -29.1556797, 30.8512211
1: -7.7828097, 28.1194229, -6.9048386, 25.4210529, -33.2038612, 35.0242615
2: -8.2539454, 28.2906685, -7.3848820, 25.5047226, -33.7586594, 35.6755524
3: -12.2706099, 28.8829918, -10.9337454, 26.0330429, -38.3036537, 39.8167343
4: -12.8631258, 28.1254616, -11.5946884, 25.2559605, -38.1190834, 39.7201462

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5267176, upper bound: 43.5667738
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5244810, upper bound: 43.5656832
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -8.3492012, 31.2999210, -37.9653130, 33.2356834
1: -7.7828097, 28.1194229, -9.8115788, 35.5474586, -43.3302612, 37.9309998
2: -8.2539454, 28.2906685, -10.3486872, 35.5690842, -43.8230286, 38.6393547
3: -12.2706099, 28.8829918, -15.4028683, 36.7234039, -48.9940109, 44.2858582
4: -12.8631258, 28.1254616, -16.3597260, 35.0432816, -47.9064064, 44.4851875

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5267176, upper bound: 43.5667738
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5244810, upper bound: 43.5656832
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -5.9647384, 22.4902878, -31.6018753, 39.8772774
1: -10.7727814, 38.5158424, -6.9048386, 25.4210529, -36.1938324, 45.4206810
2: -11.2928276, 38.6079216, -7.3848820, 25.5047226, -36.7975464, 45.9928055
3: -16.8676147, 39.8401985, -10.9337454, 26.0330429, -42.9006577, 50.7739410
4: -17.7444897, 38.1677361, -11.5946884, 25.2559605, -43.0004463, 49.7624207

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165332, upper bound: 43.5553002
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -8.3492012, 31.2999210, -40.4115067, 42.2617416
1: -10.7727814, 38.5158424, -9.8115788, 35.5474586, -46.3202400, 48.3274231
2: -11.2928276, 38.6079216, -10.3486872, 35.5690842, -46.8619118, 48.9566078
3: -16.8676147, 39.8401985, -15.4028683, 36.7234039, -53.5910187, 55.2430649
4: -17.7444897, 38.1677361, -16.3597260, 35.0432816, -52.7877731, 54.5274620

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165332, upper bound: 43.5553002
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -6.5063033, 24.2676392, -30.9330311, 31.3927879
1: -7.7828097, 28.1194229, -7.5842962, 27.4015598, -35.1843605, 35.7037201
2: -8.2539454, 28.2906685, -8.0552320, 27.5818462, -35.8357849, 36.3459015
3: -12.2706099, 28.8829918, -11.9514847, 28.1189060, -40.3895111, 40.8344765
4: -12.8631258, 28.1254616, -12.5255804, 27.4433079, -40.3064308, 40.6510429

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259161, upper bound: 43.5555644
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5234224, upper bound: 43.5542586
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -8.9248209, 33.1853523, -39.8507385, 33.8113060
1: -7.7828097, 28.1194229, -10.5382071, 37.6687279, -45.4515343, 38.6576309
2: -8.2539454, 28.2906685, -11.0602732, 37.7767830, -46.0307274, 39.3509407
3: -12.2706099, 28.8829918, -16.4940701, 38.9501419, -51.2207527, 45.3770599
4: -12.8631258, 28.1254616, -17.3545589, 37.3720055, -50.2351265, 45.4800186

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259161, upper bound: 43.5555644
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5234224, upper bound: 43.5542586
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -6.5063033, 24.2676392, -33.3792267, 40.4188423
1: -10.7727814, 38.5158424, -7.5842962, 27.4015598, -38.1743393, 46.1001358
2: -11.2928276, 38.6079216, -8.0552320, 27.5818462, -38.8746719, 46.6631546
3: -16.8676147, 39.8401985, -11.9514847, 28.1189060, -44.9865189, 51.7916832
4: -17.7444897, 38.1677361, -12.5255804, 27.4433079, -45.1877975, 50.6933174

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -8.9248209, 33.1853523, -42.2969360, 42.8373604
1: -10.7727814, 38.5158424, -10.5382071, 37.6687279, -48.4415092, 49.0540466
2: -11.2928276, 38.6079216, -11.0602732, 37.7767830, -49.0696106, 49.6681938
3: -16.8676147, 39.8401985, -16.4940701, 38.9501419, -55.8177567, 56.3342667
4: -17.7444897, 38.1677361, -17.3545589, 37.3720055, -55.1164932, 55.5222931

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -6.1316075, 23.1411781, -29.2727814, 29.2727814
1: -7.1131840, 26.1596413, -7.1131840, 26.1596413, -33.2728233, 33.2728233
2: -7.5950131, 26.2545033, -7.5950131, 26.2545033, -33.8495140, 33.8495140
3: -11.2687273, 26.8291836, -11.2687273, 26.8291836, -38.0979080, 38.0979080
4: -11.9468098, 25.9775772, -11.9468098, 25.9775772, -37.9243813, 37.9243851

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5689896, upper bound: 43.5476815
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5512625, upper bound: 43.5434515
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -8.5328178, 32.0199852, -38.1515923, 31.6739922
1: -7.1131840, 26.1596413, -10.0470028, 36.3811836, -43.4943695, 36.2066422
2: -7.5950131, 26.2545033, -10.5810204, 36.3961143, -43.9911270, 36.8355217
3: -11.2687273, 26.8291836, -15.7773991, 37.6067963, -48.8755226, 42.6065826
4: -11.9468098, 25.9775772, -16.7466125, 35.8476906, -47.7945023, 42.7241898

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5689896, upper bound: 43.5476815
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5512625, upper bound: 43.5434515
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -6.1316075, 23.1411781, -31.6739883, 38.1515923
1: -10.0470028, 36.3811836, -7.1131840, 26.1596413, -36.2066422, 43.4943695
2: -10.5810204, 36.3961143, -7.5950131, 26.2545033, -36.8355217, 43.9911270
3: -15.7773991, 37.6067963, -11.2687273, 26.8291836, -42.6065826, 48.8755226
4: -16.7466125, 35.8476906, -11.9468098, 25.9775772, -42.7241898, 47.7945023

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5395222, upper bound: 43.5233726
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5221274, upper bound: 43.5221274
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -8.5328178, 32.0199852, -40.5528030, 40.5528030
1: -10.0470028, 36.3811836, -10.0470028, 36.3811836, -46.4281845, 46.4281845
2: -10.5810204, 36.3961143, -10.5810204, 36.3961143, -46.9771347, 46.9771347
3: -15.7773991, 37.6067963, -15.7773991, 37.6067963, -53.3841934, 53.3841934
4: -16.7466125, 35.8476906, -16.7466125, 35.8476906, -52.5943031, 52.5943031

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5395222, upper bound: 43.5233726
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5221274, upper bound: 43.5221274
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -6.6653934, 24.8864861, -31.0180874, 29.8065701
1: -7.1131840, 26.1596413, -7.7828097, 28.1194229, -35.2326050, 33.9424477
2: -7.5950131, 26.2545033, -8.2539454, 28.2906685, -35.8856812, 34.5084457
3: -11.2687273, 26.8291836, -12.2706099, 28.8829918, -40.1517181, 39.0997849
4: -11.9468098, 25.9775772, -12.8631258, 28.1254616, -40.0722733, 38.8406982

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5669459, upper bound: 43.5224054
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5492188, upper bound: 43.5181754
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -9.1115875, 33.9125404, -40.0441475, 32.2527657
1: -7.1131840, 26.1596413, -10.7727814, 38.5158424, -45.6290283, 36.9324226
2: -7.5950131, 26.2545033, -11.2928276, 38.6079216, -46.2029343, 37.5473328
3: -11.2687273, 26.8291836, -16.8676147, 39.8401985, -51.1089249, 43.6967926
4: -11.9468098, 25.9775772, -17.7444897, 38.1677361, -50.1145477, 43.7220612

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5669459, upper bound: 43.5224054
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5492188, upper bound: 43.5181754
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -6.6653934, 24.8864861, -33.4192963, 38.6853790
1: -10.0470028, 36.3811836, -7.7828097, 28.1194229, -38.1664276, 44.1639900
2: -10.5810204, 36.3961143, -8.2539454, 28.2906685, -38.8716888, 44.6500587
3: -15.7773991, 37.6067963, -12.2706099, 28.8829918, -44.6603928, 49.8774071
4: -16.7466125, 35.8476906, -12.8631258, 28.1254616, -44.8720741, 48.7108154

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213290, upper bound: 43.5142460
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.5328178, 32.0199852, -9.1115875, 33.9125404, -42.4453583, 41.1315727
1: -10.0470028, 36.3811836, -10.7727814, 38.5158424, -48.5628433, 47.1539650
2: -10.5810204, 36.3961143, -11.2928276, 38.6079216, -49.1889420, 47.6889420
3: -15.7773991, 37.6067963, -16.8676147, 39.8401985, -55.6175995, 54.4744110
4: -16.7466125, 35.8476906, -17.7444897, 38.1677361, -54.9143486, 53.5921783

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213290, upper bound: 43.5142460
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -6.1316075, 23.1411781, -29.8065701, 31.0180874
1: -7.7828097, 28.1194229, -7.1131840, 26.1596413, -33.9424477, 35.2326050
2: -8.2539454, 28.2906685, -7.5950131, 26.2545033, -34.5084457, 35.8856812
3: -12.2706099, 28.8829918, -11.2687273, 26.8291836, -39.0997887, 40.1517181
4: -12.8631258, 28.1254616, -11.9468098, 25.9775772, -38.8407021, 40.0722733

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5228942, upper bound: 43.5429578
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5204005, upper bound: 43.5416520
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -8.5328178, 32.0199852, -38.6853790, 33.4192963
1: -7.7828097, 28.1194229, -10.0470028, 36.3811836, -44.1639900, 38.1664276
2: -8.2539454, 28.2906685, -10.5810204, 36.3961143, -44.6500587, 38.8716888
3: -12.2706099, 28.8829918, -15.7773991, 37.6067963, -49.8774071, 44.6603928
4: -12.8631258, 28.1254616, -16.7466125, 35.8476906, -48.7108154, 44.8720741

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5228942, upper bound: 43.5429578
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5204005, upper bound: 43.5416520
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -6.1316075, 23.1411781, -32.2527657, 40.0441475
1: -10.7727814, 38.5158424, -7.1131840, 26.1596413, -36.9324226, 45.6290283
2: -11.2928276, 38.6079216, -7.5950131, 26.2545033, -37.5473328, 46.2029343
3: -16.8676147, 39.8401985, -11.2687273, 26.8291836, -43.6967926, 51.1089249
4: -17.7444897, 38.1677361, -11.9468098, 25.9775772, -43.7220612, 50.1145477

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -8.5328178, 32.0199852, -41.1315727, 42.4453583
1: -10.7727814, 38.5158424, -10.0470028, 36.3811836, -47.1539650, 48.5628433
2: -11.2928276, 38.6079216, -10.5810204, 36.3961143, -47.6889420, 49.1889420
3: -16.8676147, 39.8401985, -15.7773991, 37.6067963, -54.4744110, 55.6175995
4: -17.7444897, 38.1677361, -16.7466125, 35.8476906, -53.5921783, 54.9143486

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -6.6653934, 24.8864861, -31.5518780, 31.5518780
1: -7.7828097, 28.1194229, -7.7828097, 28.1194229, -35.9022331, 35.9022293
2: -8.2539454, 28.2906685, -8.2539454, 28.2906685, -36.5446129, 36.5446129
3: -12.2706099, 28.8829918, -12.2706099, 28.8829918, -41.1536026, 41.1536026
4: -12.8631258, 28.1254616, -12.8631258, 28.1254616, -40.9885864, 40.9885864

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208505, upper bound: 43.5176817
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5183569, upper bound: 43.5163758
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -9.1115875, 33.9125404, -40.5779343, 33.9980736
1: -7.7828097, 28.1194229, -10.7727814, 38.5158424, -46.2986450, 38.8922043
2: -8.2539454, 28.2906685, -11.2928276, 38.6079216, -46.8618660, 39.5834961
3: -12.2706099, 28.8829918, -16.8676147, 39.8401985, -52.1108055, 45.7506065
4: -12.8631258, 28.1254616, -17.7444897, 38.1677361, -51.0308609, 45.8699493

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208505, upper bound: 43.5176817
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5183569, upper bound: 43.5163758
time: 0.46 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -6.6653934, 24.8864861, -33.9980736, 40.5779343
1: -10.7727814, 38.5158424, -7.7828097, 28.1194229, -38.8922043, 46.2986450
2: -11.2928276, 38.6079216, -8.2539454, 28.2906685, -39.5834961, 46.8618660
3: -16.8676147, 39.8401985, -12.2706099, 28.8829918, -45.7506065, 52.1108093
4: -17.7444897, 38.1677361, -12.8631258, 28.1254616, -45.8699493, 51.0308609

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -9.1115875, 33.9125404, -43.0241280, 43.0241280
1: -10.7727814, 38.5158424, -10.7727814, 38.5158424, -49.2886238, 49.2886238
2: -11.2928276, 38.6079216, -11.2928276, 38.6079216, -49.9007492, 49.9007492
3: -16.8676147, 39.8401985, -16.8676147, 39.8401985, -56.7078133, 56.7078133
4: -17.7444897, 38.1677361, -17.7444897, 38.1677361, -55.9122238, 55.9122238

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.91 + 373.86 = 376.77 seconds
