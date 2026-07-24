## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2954094305


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5784860, 0.5784855)
1: (-16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8586702, 0.8586698)
2: (-4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6296978, 0.6296978)
3: (-12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6798048, 0.6798053)
4: (-10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5727429, 0.5727427)
5: (-7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6351314, 0.6351314)
6: (-5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9446011, 0.9446011)
7: (-11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8562999, 0.8562999)
8: (-2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5692954, 0.5692954)
9: (-2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6717248, 0.6717248)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.96 + 34.67 = 56.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2968933, upper bound: 0.2968939

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 554

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910394, upper bound: 0.2963058
time: 4.91 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968893, upper bound: 0.2968916
time: 9.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.68
Output dim: 0, lower bound: -0.2910394, upper bound: 0.2963058
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.68
Output dim: 0, lower bound: -0.2968893, upper bound: 0.2968916

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 10.2774029, 11.2591724, 10.2575979, 11.2680807, -0.5365911, 0.5449851
1: -16.7288456, -15.2674465, -16.7328491, -15.2654371, -0.8512936, 0.8492455
2: -4.6773100, -3.6597071, -4.6836247, -3.6549640, -0.6156864, 0.6126146
3: -12.7141876, -11.6603441, -12.7173424, -11.6536055, -0.6690178, 0.6662149
4: -10.3716249, -9.2244844, -10.3773499, -9.2131805, -0.5520501, 0.5485330
5: -7.7624745, -6.6608458, -7.7666683, -6.6591530, -0.6257734, 0.6289787
6: -5.3930507, -4.3150768, -5.4079771, -4.3095865, -0.9157438, 0.9252462
7: -11.3011446, -9.8888874, -11.3042183, -9.8824558, -0.8473425, 0.8436093
8: -2.8602200, -1.9520285, -2.8611026, -1.9507697, -0.5665379, 0.5661850
9: -2.4807153, -1.2665973, -2.4878013, -1.2591164, -0.6522708, 0.6533465

Time for backsubstitution: 20.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2910358, upper bound: 0.2933882
time: 5.15 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910376, upper bound: 0.2963039
time: 4.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 10.2391586, 11.2704563, 10.2391586, 11.2704544, -0.5462122, 0.5782003
1: -16.7365799, -15.2644939, -16.7365837, -15.2644939, -0.8634281, 0.8567891
2: -4.6894836, -3.6523602, -4.6894846, -3.6523633, -0.6258740, 0.6284761
3: -12.7181883, -11.6473265, -12.7181892, -11.6473255, -0.6794925, 0.6704516
4: -10.3790951, -9.2027006, -10.3790951, -9.2026939, -0.5720475, 0.5542607
5: -7.7704864, -6.6585093, -7.7704878, -6.6585097, -0.6332593, 0.6351318
6: -5.4215250, -4.3090539, -5.4215269, -4.3090539, -0.9232912, 0.9445992
7: -11.3050499, -9.8765717, -11.3050499, -9.8765707, -0.8562999, 0.8486710
8: -2.8618193, -1.9497166, -2.8618202, -1.9497166, -0.5712228, 0.5692945
9: -2.4918127, -1.2521224, -2.4918118, -1.2521186, -0.6711285, 0.6602845

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2963060, upper bound: 0.2910394
time: 8.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2963060, upper bound: 0.2910393
time: 8.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 39.07 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 39.07
Output dim: 0, lower bound: -0.2910358, upper bound: 0.2933882
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 39.07
Output dim: 0, lower bound: -0.2910376, upper bound: 0.2963039
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 39.07
Output dim: 0, lower bound: -0.2963060, upper bound: 0.2910394
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 39.07
Output dim: 0, lower bound: -0.2963060, upper bound: 0.2910393

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 10.2774048, 11.2591724, 10.2576008, 11.2680798, -0.5363927, 0.5318584
1: -16.7288513, -15.2674427, -16.7328453, -15.2654362, -0.8499789, 0.8512540
2: -4.6773100, -3.6597099, -4.6836247, -3.6549690, -0.6028028, 0.6120691
3: -12.7141876, -11.6603422, -12.7173376, -11.6536112, -0.6558757, 0.6658983
4: -10.3716249, -9.2244844, -10.3773489, -9.2131805, -0.5493505, 0.5419848
5: -7.7624707, -6.6608438, -7.7666674, -6.6591539, -0.6251135, 0.5993769
6: -5.3930502, -4.3150778, -5.4079752, -4.3095884, -0.9157438, 0.9050808
7: -11.3011436, -9.8888874, -11.3042183, -9.8824577, -0.8422742, 0.8436065
8: -2.8602209, -1.9520309, -2.8611035, -1.9507740, -0.5637217, 0.5661836
9: -2.4807148, -1.2665975, -2.4878004, -1.2591174, -0.6486950, 0.6532898

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6143

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2895567, upper bound: 0.2963019
time: 6.29 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910350, upper bound: 0.2963014
time: 4.26 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 10.2393723, 11.2704239, 10.2774029, 11.2591724, -0.5465827, 0.5394075
1: -16.7365837, -15.2645073, -16.7288456, -15.2674465, -0.8535938, 0.8504705
2: -4.6894131, -3.6524003, -4.6773100, -3.6597071, -0.6168349, 0.6176744
3: -12.7181816, -11.6474981, -12.7141876, -11.6603441, -0.6667986, 0.6751795
4: -10.3790684, -9.2027483, -10.3716249, -9.2244844, -0.5506697, 0.5521750
5: -7.7704759, -6.6585221, -7.7624745, -6.6608458, -0.6332917, 0.6261044
6: -5.4214048, -4.3090563, -5.3930507, -4.3150768, -0.9308043, 0.9161577
7: -11.3050385, -9.8767681, -11.3011446, -9.8888874, -0.8440619, 0.8530040
8: -2.8618073, -1.9497182, -2.8602200, -1.9520285, -0.5668206, 0.5678287
9: -2.4917541, -1.2522111, -2.4807153, -1.2665973, -0.6568880, 0.6573749

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2933861, upper bound: 0.2910358
time: 7.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2963016, upper bound: 0.2910375
time: 8.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 10.2391586, 11.2704563, 10.2391586, 11.2704563, -0.5462122, 0.5462124
1: -16.7365799, -15.2644939, -16.7365799, -15.2644939, -0.8634257, 0.8634262
2: -4.6894836, -3.6523602, -4.6894836, -3.6523602, -0.6258731, 0.6258731
3: -12.7181883, -11.6473265, -12.7181883, -11.6473265, -0.6704507, 0.6704507
4: -10.3790951, -9.2027006, -10.3790951, -9.2027006, -0.5542605, 0.5542605
5: -7.7704864, -6.6585093, -7.7704864, -6.6585093, -0.6332593, 0.6332593
6: -5.4215250, -4.3090539, -5.4215250, -4.3090539, -0.9232912, 0.9232912
7: -11.3050499, -9.8765717, -11.3050499, -9.8765717, -0.8486700, 0.8486705
8: -2.8618193, -1.9497166, -2.8618193, -1.9497166, -0.5712223, 0.5712223
9: -2.4918127, -1.2521224, -2.4918127, -1.2521224, -0.6602838, 0.6602836

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2933883, upper bound: 0.2916537
time: 6.05 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2963039, upper bound: 0.2910376
time: 5.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.01 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.01
Output dim: 0, lower bound: -0.2895567, upper bound: 0.2963019
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.01
Output dim: 0, lower bound: -0.2910350, upper bound: 0.2963014
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 34.01
Output dim: 0, lower bound: -0.2933861, upper bound: 0.2910358
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 34.01
Output dim: 0, lower bound: -0.2963016, upper bound: 0.2910375
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 34.01
Output dim: 0, lower bound: -0.2933883, upper bound: 0.2916537
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.01
Output dim: 0, lower bound: -0.2963039, upper bound: 0.2910376

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 10.2854910, 11.2537336, 10.2622271, 11.2679596, -0.5281305, 0.5201705
1: -16.7266998, -15.2688293, -16.7318039, -15.2655535, -0.8479204, 0.8482637
2: -4.6673203, -3.6660373, -4.6779141, -3.6551359, -0.5929694, 0.5982394
3: -12.7103930, -11.6663837, -12.7168827, -11.6571465, -0.6484995, 0.6594534
4: -10.3657665, -9.2287359, -10.3739176, -9.2136889, -0.5431943, 0.5347590
5: -7.7474928, -6.6838789, -7.7661633, -6.6728086, -0.5919650, 0.5758178
6: -5.3819518, -4.3215117, -5.4016767, -4.3100209, -0.9048719, 0.8927937
7: -11.2987700, -9.8907881, -11.3030233, -9.8830242, -0.8395729, 0.8402467
8: -2.8567934, -1.9561758, -2.8606024, -1.9532061, -0.5581808, 0.5617337
9: -2.4721625, -1.2715409, -2.4830308, -1.2593431, -0.6398430, 0.6436005

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6143
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5831

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2894126, upper bound: 0.2944378
time: 6.19 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2895560, upper bound: 0.2963000
time: 4.95 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 10.2774124, 11.2591724, 10.2576065, 11.2680798, -0.5261831, 0.5268292
1: -16.7288456, -15.2674427, -16.7328434, -15.2654371, -0.8497534, 0.8520017
2: -4.6773009, -3.6597080, -4.6836195, -3.6549683, -0.5908871, 0.6064286
3: -12.7141867, -11.6603479, -12.7173367, -11.6536102, -0.6558743, 0.6626453
4: -10.3716221, -9.2244854, -10.3773479, -9.2131805, -0.5440609, 0.5419834
5: -7.7624741, -6.6608639, -7.7666674, -6.6591659, -0.6111307, 0.5688374
6: -5.3930454, -4.3150806, -5.4079723, -4.3095870, -0.9074602, 0.9035988
7: -11.3011417, -9.8888893, -11.3042183, -9.8824558, -0.8421550, 0.8439970
8: -2.8602185, -1.9520326, -2.8611035, -1.9507747, -0.5636911, 0.5618792
9: -2.4807103, -1.2665963, -2.4877958, -1.2591171, -0.6398194, 0.6532879

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6143
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5831

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2908908, upper bound: 0.2944381
time: 5.34 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910342, upper bound: 0.2962995
time: 6.39 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 10.2393770, 11.2704220, 10.2774048, 11.2591724, -0.5334573, 0.5392032
1: -16.7365799, -15.2645140, -16.7288513, -15.2674427, -0.8556027, 0.8491549
2: -4.6894121, -3.6524043, -4.6773100, -3.6597099, -0.6122589, 0.6047902
3: -12.7181797, -11.6475019, -12.7141876, -11.6603422, -0.6664796, 0.6620054
4: -10.3790627, -9.2027502, -10.3716249, -9.2244844, -0.5440969, 0.5494485
5: -7.7704782, -6.6585236, -7.7624707, -6.6608438, -0.6036913, 0.6246397
6: -5.4214001, -4.3090572, -5.3930502, -4.3150778, -0.9101019, 0.9161568
7: -11.3050365, -9.8767710, -11.3011436, -9.8888874, -0.8440590, 0.8479357
8: -2.8618073, -1.9497199, -2.8602209, -1.9520309, -0.5668192, 0.5650125
9: -2.4917548, -1.2522111, -2.4807148, -1.2665975, -0.6568241, 0.6535420

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6143

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2962991, upper bound: 0.2895566
time: 4.19 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2962991, upper bound: 0.2910349
time: 6.43 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 10.2391644, 11.2704544, 10.2391605, 11.2704582, -0.5331242, 0.5460079
1: -16.7365799, -15.2644892, -16.7365799, -15.2644911, -0.8654461, 0.8621116
2: -4.6894808, -3.6523664, -4.6894817, -3.6523659, -0.6258717, 0.6129899
3: -12.7181883, -11.6473322, -12.7181892, -11.6473284, -0.6701326, 0.6572924
4: -10.3790903, -9.2026997, -10.3790922, -9.2027006, -0.5476968, 0.5541232
5: -7.7704873, -6.6585093, -7.7704864, -6.6585116, -0.6036580, 0.6332579
6: -5.4215231, -4.3090549, -5.4215236, -4.3090553, -0.9031305, 0.9232922
7: -11.3050451, -9.8765745, -11.3050499, -9.8765726, -0.8486691, 0.8436027
8: -2.8618202, -1.9497201, -2.8618188, -1.9497178, -0.5712209, 0.5684075
9: -2.4918120, -1.2521248, -2.4918113, -1.2521241, -0.6602201, 0.6567378

Time for backsubstitution: 20.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6143

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968848, upper bound: 0.2901698
time: 5.20 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968847, upper bound: 0.2916511
time: 6.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.11 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2894126, upper bound: 0.2944378
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2895560, upper bound: 0.2963000
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2908908, upper bound: 0.2944381
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2910342, upper bound: 0.2962995
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2962991, upper bound: 0.2895566
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2962991, upper bound: 0.2910349
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2968848, upper bound: 0.2901698
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.11
Output dim: 0, lower bound: -0.2968847, upper bound: 0.2916511

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 10.2854919, 11.2537336, 10.2622337, 11.2679577, -0.5281277, 0.5092132
1: -16.7266960, -15.2688313, -16.7318039, -15.2655563, -0.8402495, 0.8482633
2: -4.6673203, -3.6660402, -4.6779127, -3.6551397, -0.5899019, 0.5962889
3: -12.7103939, -11.6663837, -12.7168808, -11.6571512, -0.6467171, 0.6589804
4: -10.3657675, -9.2287359, -10.3739185, -9.2136908, -0.5334218, 0.5347555
5: -7.7474928, -6.6838827, -7.7661610, -6.6728082, -0.5895674, 0.5699871
6: -5.3819504, -4.3215113, -5.4016771, -4.3100228, -0.9054508, 0.8919582
7: -11.2987652, -9.8907881, -11.3030167, -9.8830252, -0.8387833, 0.8318162
8: -2.8567944, -1.9561753, -2.8606005, -1.9532082, -0.5535088, 0.5617328
9: -2.4721608, -1.2715437, -2.4830301, -1.2593467, -0.6381021, 0.6431503

Time for backsubstitution: 20.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2866375, upper bound: 0.2944370
time: 11.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2866375, upper bound: 0.2933836
time: 5.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 10.2774134, 11.2591724, 10.2576084, 11.2680798, -0.5261812, 0.5158718
1: -16.7288475, -15.2674446, -16.7328453, -15.2654390, -0.8420825, 0.8519998
2: -4.6773009, -3.6597109, -4.6836190, -3.6549709, -0.5878191, 0.6044774
3: -12.7141867, -11.6603470, -12.7173376, -11.6536121, -0.6540909, 0.6621723
4: -10.3716230, -9.2244873, -10.3773470, -9.2131815, -0.5342879, 0.5419803
5: -7.7624722, -6.6608677, -7.7666631, -6.6591659, -0.6087339, 0.5630050
6: -5.3930430, -4.3150787, -5.4079719, -4.3095894, -0.9080410, 0.9026151
7: -11.3011370, -9.8888884, -11.3042088, -9.8824577, -0.8413653, 0.8355675
8: -2.8602180, -1.9520340, -2.8611040, -1.9507751, -0.5590200, 0.5618787
9: -2.4807091, -1.2666006, -2.4877987, -1.2591193, -0.6380796, 0.6528382

Time for backsubstitution: 20.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2881155, upper bound: 0.2962994
time: 7.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2881155, upper bound: 0.2933836
time: 5.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 10.2440014, 11.2702942, 10.2854910, 11.2537336, -0.5217726, 0.5309353
1: -16.7355423, -15.2646313, -16.7266998, -15.2688293, -0.8526263, 0.8470945
2: -4.6837025, -3.6525776, -4.6673203, -3.6660373, -0.5984392, 0.5949526
3: -12.7177181, -11.6510363, -12.7103930, -11.6663837, -0.6600294, 0.6546302
4: -10.3756371, -9.2032566, -10.3657665, -9.2287359, -0.5368769, 0.5432916
5: -7.7699738, -6.6721773, -7.7474928, -6.6838789, -0.5801313, 0.5914931
6: -5.4151058, -4.3094931, -5.3819518, -4.3215117, -0.8959875, 0.9052830
7: -11.3038483, -9.8773384, -11.2987700, -9.8907881, -0.8406997, 0.8452349
8: -2.8612995, -1.9521525, -2.8567934, -1.9561758, -0.5623722, 0.5594745
9: -2.4869926, -1.2524395, -2.4721625, -1.2715409, -0.6471415, 0.6447072

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2944357, upper bound: 0.2894131
time: 5.18 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2962972, upper bound: 0.2895566
time: 6.04 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 10.2393789, 11.2704239, 10.2774124, 11.2591724, -0.5284276, 0.5289772
1: -16.7365799, -15.2645111, -16.7288456, -15.2674427, -0.8563504, 0.8489294
2: -4.6894078, -3.6524036, -4.6773009, -3.6597080, -0.6066182, 0.5928745
3: -12.7181778, -11.6475029, -12.7141867, -11.6603479, -0.6632271, 0.6618741
4: -10.3790617, -9.2027493, -10.3716221, -9.2244854, -0.5440960, 0.5441585
5: -7.7704763, -6.6585321, -7.7624741, -6.6608639, -0.5731514, 0.6106575
6: -5.4214010, -4.3090572, -5.3930454, -4.3150806, -0.9047999, 0.9078751
7: -11.3050375, -9.8767700, -11.3011417, -9.8888893, -0.8444495, 0.8478169
8: -2.8618064, -1.9497211, -2.8602185, -1.9520326, -0.5625067, 0.5649815
9: -2.4917488, -1.2522123, -2.4807103, -1.2665963, -0.6568217, 0.6445105

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2944357, upper bound: 0.2908909
time: 5.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2962972, upper bound: 0.2910348
time: 7.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 10.2437887, 11.2703276, 10.2472363, 11.2650003, -0.5225215, 0.5377555
1: -16.7355423, -15.2646103, -16.7344208, -15.2658806, -0.8624616, 0.8600488
2: -4.6837749, -3.6525402, -4.6794968, -3.6587036, -0.6121488, 0.6031661
3: -12.7177277, -11.6508656, -12.7143898, -11.6533594, -0.6637049, 0.6498995
4: -10.3756647, -9.2032080, -10.3732595, -9.2069521, -0.5404823, 0.5479815
5: -7.7699838, -6.6721649, -7.7554941, -6.6815314, -0.5801034, 0.6004894
6: -5.4152293, -4.3094897, -5.4104424, -4.3154941, -0.8908329, 0.9124136
7: -11.3038578, -9.8771410, -11.3026886, -9.8784714, -0.8453097, 0.8409133
8: -2.8613138, -1.9521494, -2.8583670, -1.9538555, -0.5667782, 0.5628529
9: -2.4870501, -1.2523525, -2.4833145, -1.2570829, -0.6505344, 0.6479278

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2900195
time: 8.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968829, upper bound: 0.2901688
time: 8.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 10.2391663, 11.2704544, 10.2391701, 11.2704544, -0.5330410, 0.5358024
1: -16.7365780, -15.2644901, -16.7365799, -15.2644901, -0.8661919, 0.8618855
2: -4.6894774, -3.6523674, -4.6894755, -3.6523669, -0.6203237, 0.6010742
3: -12.7181883, -11.6473322, -12.7181873, -11.6473312, -0.6668797, 0.6572905
4: -10.3790894, -9.2026987, -10.3790913, -9.2026987, -0.5476952, 0.5493765
5: -7.7704868, -6.6585226, -7.7704859, -6.6585317, -0.5731180, 0.6196523
6: -5.4215183, -4.3090549, -5.4215202, -4.3090534, -0.9031239, 0.9150085
7: -11.3050461, -9.8765726, -11.3050470, -9.8765736, -0.8490591, 0.8434830
8: -2.8618202, -1.9497206, -2.8618202, -1.9497213, -0.5669088, 0.5683861
9: -2.4918075, -1.2521217, -2.4918056, -1.2521234, -0.6602187, 0.6478615

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 123

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2950164, upper bound: 0.2915013
time: 5.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968851, upper bound: 0.2916504
time: 16.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 43.87 seconds
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2866375, upper bound: 0.2944370
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2866375, upper bound: 0.2933836
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2881155, upper bound: 0.2962994
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2881155, upper bound: 0.2933836
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2944357, upper bound: 0.2894131
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2962972, upper bound: 0.2895566
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2944357, upper bound: 0.2908909
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2962972, upper bound: 0.2910348
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2900195
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2968829, upper bound: 0.2901688
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2950164, upper bound: 0.2915013
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 43.87
Output dim: 0, lower bound: -0.2968851, upper bound: 0.2916504

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 10.2988319, 11.2546425, 10.2577314, 11.2680788, -0.5037527, 0.5115144
1: -16.7263870, -15.2686520, -16.7326260, -15.2654762, -0.8396707, 0.8470030
2: -4.6719127, -3.6735940, -4.6836190, -3.6550202, -0.5899005, 0.5905092
3: -12.7087107, -11.6744261, -12.7173347, -11.6537943, -0.6563458, 0.6481261
4: -10.3636532, -9.2276468, -10.3769321, -9.2132015, -0.5260651, 0.5411165
5: -7.7298312, -6.6714716, -7.7666359, -6.6592059, -0.5753131, 0.5645423
6: -5.3655753, -4.3246698, -5.4078693, -4.3098416, -0.8812656, 0.8961802
7: -11.2936792, -9.8963900, -11.3039207, -9.8824759, -0.8393121, 0.8277063
8: -2.8583865, -1.9606929, -2.8610759, -1.9511175, -0.5596681, 0.5526557
9: -2.4772305, -1.2703154, -2.4876881, -1.2591333, -0.6379890, 0.6491733

Time for backsubstitution: 22.16 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.63 + 564.28 = 620.92 seconds
