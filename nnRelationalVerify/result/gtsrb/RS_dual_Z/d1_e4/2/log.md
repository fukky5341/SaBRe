## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 1800 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9000626, 13.9000664)
1: (-3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5087318)
2: (-0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614334, 13.4614372)
3: (-1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319748, 12.0319729)
4: (-11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6887894, 14.6887856)
5: (1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796)
6: (-39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2155037, 15.2154999)
7: (-3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6391602, 13.6391640)
8: (-6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1128159, 12.1128178)
9: (-4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0374336, 13.0374336)
10: (1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535446, 20.9535446)
11: (-11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476)
12: (-11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0364494, 15.0364494)
13: (-18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293907, 16.6293945)
14: (4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7749405, 26.7749405)
15: (-8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198)
16: (-16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331146, 14.8331184)
17: (6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2550125, 17.2550125)
18: (-14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4317722, 14.4317741)
19: (-20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5550842, 14.5550919)
20: (-2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6365433, 12.6365433)
21: (-11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549)
22: (-3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9717941, 14.9717941)
23: (-14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3402557, 14.3402557)
24: (-19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2769775, 9.2769775)
25: (-5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234940, 13.8234940)
26: (-21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3724327, 19.3724365)
27: (-16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2311211, 13.2311211)
28: (-12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141)
29: (-5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9777222, 14.9777222)
30: (-10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620842, 13.5620842)
31: (-10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6624298, 14.6624260)
32: (-24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3205376, 13.3205414)
33: (-69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6724014, 16.6724014)
34: (-53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1694450, 14.1694450)
35: (-47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0219498, 13.0219536)
36: (-42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1241531, 15.1241570)
37: (-86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9252739, 18.9252701)
38: (-52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3915100, 18.3915100)
39: (-76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0968819, 16.0968819)
40: (-67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520813, 14.3520813)
41: (-55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7036438, 16.7036476)
42: (-29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716637, 17.2716675)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.73 + 21.49 = 24.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -12.5728910, upper bound: 12.5728909

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5632410, upper bound: 12.5658605
time: 9.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658605, upper bound: 12.5632410
time: 25.22 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 35.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 35.04
Output dim: 14, lower bound: -12.5632410, upper bound: 12.5658605
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 35.04
Output dim: 14, lower bound: -12.5658605, upper bound: 12.5632410

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8994942, 13.9010925
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5085602, 8.5089664
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4611969, 13.4619293
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0315781, 12.0325279
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6902466, 14.6886292
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2156944, 15.2148972
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6388702, 13.6396561
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1122208, 12.1138744
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0370560, 13.0379562
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9527130, 20.9549255
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0365791, 15.0362129
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6313019, 16.6292191
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7742538, 26.7760086
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8322906, 14.8345299
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2538719, 17.2563286
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4319992, 14.4316139
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5555878, 14.5547180
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6372757, 12.6360855
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9731903, 14.9710579
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3401489, 14.3414001
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2775345, 9.2766685
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234940, 13.8234863
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3729515, 19.3719788
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2317505, 13.2307587
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9780731, 14.9775085
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620689, 13.5621185
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6627579, 14.6621780
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3210945, 13.3198738
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6748848, 16.6708527
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1713486, 14.1682014
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0246849, 13.0204277
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1268539, 15.1220894
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9269791, 18.9241409
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3949242, 18.3889694
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.1002502, 16.0946503
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520126, 14.3520107
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7043762, 16.7029076
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2715340, 17.2730713

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5483765, upper bound: 12.5652900
time: 10.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5627228, upper bound: 12.5513893
time: 8.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9000626, 13.8994942
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5085621
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614334, 13.4611931
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319748, 12.0315781
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6886292, 14.6887856
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2149010, 15.2154999
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6391602, 13.6388702
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1128159, 12.1122227
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0374336, 13.0370560
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535446, 20.9527130
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0362129, 15.0364494
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6292191, 16.6293945
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7749405, 26.7742462
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331146, 14.8322945
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2550125, 17.2538719
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4316139, 14.4317741
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5547180, 14.5550919
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6360855, 12.6365433
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9710541, 14.9717941
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3402557, 14.3401489
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2766685, 9.2769775
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234863, 13.8234940
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3719749, 19.3724365
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2307587, 13.2311211
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9775085, 14.9777222
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620842, 13.5620689
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6621780, 14.6624260
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3198738, 13.3205414
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6708488, 16.6724014
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1681976, 14.1694450
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0204277, 13.0219536
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1220932, 15.1241570
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9241409, 18.9252701
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3889732, 18.3915100
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0946503, 16.0968819
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520126, 14.3520813
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7029037, 16.7036476
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716637, 17.2715302

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5513893, upper bound: 12.5627227
time: 11.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5652901, upper bound: 12.5483765
time: 9.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.45
Output dim: 14, lower bound: -12.5483765, upper bound: 12.5652900
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.45
Output dim: 14, lower bound: -12.5627228, upper bound: 12.5513893
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.45
Output dim: 14, lower bound: -12.5513893, upper bound: 12.5627227
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.45
Output dim: 14, lower bound: -12.5652901, upper bound: 12.5483765

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8925629, 13.8949203
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5089874, 8.5093498
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4589157, 13.4599380
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0285721, 12.0296688
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6930618, 14.6902771
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2034225, 15.2012825
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6390152, 13.6397858
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1052742, 12.1075974
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0284805, 13.0302544
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9406891, 20.9446411
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0346375, 15.0341301
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6335907, 16.6310844
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7636261, 26.7666931
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8209305, 14.8240013
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2383919, 17.2426567
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4265652, 14.4254646
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5507202, 14.5491295
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6297073, 12.6275330
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9680061, 14.9649391
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3411865, 14.3432312
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2732925, 9.2721977
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236847, 13.8236847
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707657, 19.3697815
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2319908, 13.2300835
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9779205, 14.9773750
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623436, 13.5624313
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6571236, 14.6557541
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3128777, 13.3105278
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6586113, 16.6523285
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1592789, 14.1544495
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0061760, 12.9994049
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1051254, 15.0975380
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9183502, 18.9145546
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3620377, 18.3517303
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0768509, 16.0680046
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518295, 14.3518772
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6961670, 16.6935616
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2774124, 17.2799377

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5475464, upper bound: 12.5484175
time: 14.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5357121, upper bound: 12.5649888
time: 9.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8933525, 13.8941612
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5089493, 8.5093880
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4592056, 13.4596558
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0287323, 12.0295200
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6919022, 14.6914902
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2020874, 15.2025452
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6389999, 13.6397972
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1059837, 12.1069298
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0293541, 13.0293808
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9425049, 20.9429016
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0344963, 15.0342712
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6331711, 16.6315079
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7649384, 26.7653809
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8217621, 14.8231621
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2402763, 17.2408485
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4258480, 14.4261799
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5500031, 14.5498428
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6287231, 12.6285362
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9670677, 14.9659119
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3419800, 14.3424339
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2730637, 9.2724342
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236923, 13.8236771
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3707657, 19.3697891
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2310753, 13.2309990
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9779358, 14.9773636
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623817, 13.5623932
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6563377, 14.6565475
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3117485, 13.3116570
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6563606, 16.6545753
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1576004, 14.1561241
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0036659, 13.0019684
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1023026, 15.1004829
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9173965, 18.9155159
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3576813, 18.3562775
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0736008, 16.0712509
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518829, 14.3518238
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6950302, 16.6947021
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2783966, 17.2789497

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5624216, upper bound: 12.5387003
time: 12.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5458459, upper bound: 12.5505590
time: 18.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8931351, 13.8933525
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5091553, 8.5089493
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591675, 13.4592056
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0289612, 12.0287342
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6914902, 14.6904335
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2025452, 15.2018890
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6393051, 13.6389999
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1058693, 12.1059799
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0288544, 13.0293541
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9415359, 20.9425049
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0342712, 15.0343704
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6315079, 16.6312599
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7643280, 26.7649384
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8217545, 14.8217659
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2395325, 17.2402725
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4261799, 14.4256248
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5498352, 14.5494919
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6285362, 12.6279945
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9659081, 14.9656792
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3413010, 14.3419838
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2724304, 9.2725182
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236771, 13.8236847
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3697891, 19.3702393
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2309990, 13.2304497
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9773636, 14.9775887
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623627, 13.5623817
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6565437, 14.6560097
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3116570, 13.3111916
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6545753, 16.6538773
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1561203, 14.1556854
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0019646, 13.0009384
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1004868, 15.0996094
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9155121, 18.9156837
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3562698, 18.3542633
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0712509, 16.0702324
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518219, 14.3519459
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6947021, 16.6943016
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2775421, 17.2783966

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5505591, upper bound: 12.5458458
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5387003, upper bound: 12.5624215
time: 7.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8939247, 13.8925629
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5091171, 8.5089874
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594498, 13.4589195
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0291290, 12.0285702
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6902847, 14.6916466
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2012863, 15.2031479
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6392975, 13.6390114
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1065712, 12.1052780
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0297279, 13.0284805
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9433441, 20.9406891
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0341301, 15.0345116
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6310806, 16.6316833
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7656403, 26.7636185
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8225937, 14.8209305
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2414169, 17.2383881
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4254627, 14.4263420
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5491333, 14.5502052
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6275330, 12.6289978
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9649391, 14.9666481
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3420944, 14.3411865
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2721977, 9.2727547
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8236847, 13.8236809
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3697891, 19.3702469
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2300835, 13.2313652
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9773788, 14.9775734
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624008, 13.5623436
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6557579, 14.6567955
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3105278, 13.3123207
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6523247, 16.6561279
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1544495, 14.1573639
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9994087, 13.0035019
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0975418, 15.1025543
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9145584, 18.9166412
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3517303, 18.3588104
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0680008, 16.0734787
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3518753, 14.3518925
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6935654, 16.6954422
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2785339, 17.2774086

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5649889, upper bound: 12.5357120
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5484176, upper bound: 12.5475463
time: 6.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5475464, upper bound: 12.5484175
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5357121, upper bound: 12.5649888
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5624216, upper bound: 12.5387003
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5458459, upper bound: 12.5505590
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5505591, upper bound: 12.5458458
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5387003, upper bound: 12.5624215
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5649889, upper bound: 12.5357120
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.39
Output dim: 14, lower bound: -12.5484176, upper bound: 12.5475463

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8891563, 13.8912468
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4987335, 8.5017166
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4597855, 13.4606018
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0343857, 12.0384560
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6815491, 14.6802406
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1238480, 15.1314774
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6167374, 13.6214981
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1067963, 12.1090908
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0037231, 13.0020332
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9508667, 20.9524078
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0241318, 15.0248985
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6300011, 16.6252937
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7370148, 26.7363586
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8226280, 14.8263321
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2373505, 17.2414780
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4225960, 14.4218597
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5474243, 14.5460434
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6223755, 12.6216316
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9556808, 14.9519310
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3182335, 14.3170242
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2767067, 9.2740173
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8151779, 13.8133965
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3329010, 19.3239288
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2362595, 13.2351227
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9576797, 14.9542732
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5637970, 13.5630836
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6548958, 14.6555214
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2792282, 13.2821960
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6275101, 16.6250381
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1157379, 14.1172638
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9921112, 12.9878502
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1006927, 15.0936584
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9269714, 18.9209747
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3232346, 18.3177032
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0781708, 16.0696106
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3451729, 14.3456612
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6816559, 16.6817093
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2739258, 17.2768440

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5330797, upper bound: 12.5481944
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5475465, upper bound: 12.5365185
time: 8.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8888893, 13.8915100
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5013542, 8.4990959
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4595871, 13.4608345
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0373993, 12.0354843
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6830215, 14.6787682
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1336136, 15.1217117
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6207199, 13.6175079
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1067734, 12.1091175
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0002556, 13.0055008
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9484558, 20.9548111
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0254135, 15.0236206
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6278038, 16.6275558
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7332916, 26.7400970
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8232841, 14.8256950
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2372131, 17.2416229
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4229622, 14.4214973
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5476379, 14.5457840
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6238022, 12.6202049
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9549942, 14.9528809
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3149757, 14.3202820
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2751122, 9.2756538
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8134003, 13.8151779
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3249054, 19.3319397
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2370300, 13.2343521
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9548111, 14.9571266
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5629959, 13.5638885
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6568947, 14.6535301
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2845421, 13.2768784
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6313248, 16.6212234
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1220932, 14.1109123
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9946213, 12.9853401
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1012497, 15.0931053
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9247742, 18.9233704
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3280106, 18.3129272
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0784607, 16.0693283
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3456230, 14.3452225
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6843109, 16.6790466
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2745743, 17.2764549

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5226077, upper bound: 12.5649890
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5356293, upper bound: 12.5505438
time: 8.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8899460, 13.8904877
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4986954, 8.5017548
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600983, 13.4603195
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0345459, 12.0383453
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6803894, 14.6814537
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1225166, 15.1327400
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6167221, 13.6215096
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1075058, 12.1084232
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0046005, 13.0011559
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9526749, 20.9506683
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0239906, 15.0250473
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6296425, 16.6257172
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7383423, 26.7350464
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8234596, 14.8255196
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2392426, 17.2396698
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4218788, 14.4225769
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5466614, 14.5467567
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6213951, 12.6226349
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9550095, 14.9528999
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3190346, 14.3162270
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2765198, 9.2742538
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8151855, 13.8133926
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3329163, 19.3239288
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2353439, 13.2360382
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9576950, 14.9542580
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5638390, 13.5630493
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6541100, 14.6563110
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2780991, 13.2833252
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6252594, 16.6272888
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1140671, 14.1189384
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9896011, 12.9904099
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0978699, 15.0966034
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9262085, 18.9219322
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3188705, 18.3222504
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0749283, 16.0728607
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3452263, 14.3456192
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6805191, 16.6828461
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2749176, 17.2761192

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5479763, upper bound: 12.5386175
time: 9.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5624217, upper bound: 12.5255980
time: 14.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8896828, 13.8907547
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5013161, 8.4991341
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598694, 13.4605255
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0375214, 12.0353355
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6818619, 14.6799812
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1322823, 15.1229744
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6207123, 13.6175194
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1074753, 12.1084499
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0011330, 13.0046234
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9502716, 20.9530792
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0252647, 15.0237656
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6273766, 16.6279144
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7346039, 26.7387848
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8240929, 14.8248558
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2390976, 17.2398148
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4222450, 14.4222145
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5469208, 14.5465508
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6228218, 12.6212082
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9540634, 14.9535904
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3157768, 14.3194847
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2748833, 9.2758446
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8134003, 13.8151703
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3249054, 19.3319244
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2361145, 13.2352676
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9548264, 14.9571152
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5630341, 13.5638504
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6561012, 14.6543198
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2834129, 13.2780075
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6290741, 16.6234703
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1204147, 14.1125908
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9921112, 12.9879036
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0984268, 15.0960464
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9238129, 18.9241333
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3236618, 18.3174744
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0752106, 16.0725746
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3456650, 14.3451691
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6831741, 16.6801872
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2753067, 17.2754669

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5339459, upper bound: 12.5505593
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5456228, upper bound: 12.5360941
time: 6.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8897285, 13.8896790
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4989014, 8.5013161
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600449, 13.4598694
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0347748, 12.0375214
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6799774, 14.6803894
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1229744, 15.1320839
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6170273, 13.6207123
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1073837, 12.1074734
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0040970, 13.0011330
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9517059, 20.9502716
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0237656, 15.0251389
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6279182, 16.6254692
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7377167, 26.7346039
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8234520, 14.8241005
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2384949, 17.2390976
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4222145, 14.4220200
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5465546, 14.5464096
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6212082, 12.6220970
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9535904, 14.9526634
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3183403, 14.3157730
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2758484, 9.2743378
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8151703, 13.8134079
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3319168, 19.3243942
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2352676, 13.2354965
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9571152, 14.9544907
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5638161, 13.5630341
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6543159, 14.6557617
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2780075, 13.2828636
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6234741, 16.6265869
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1125870, 14.1184998
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9879074, 12.9893723
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0960464, 15.0957260
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9241333, 18.9221001
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3174667, 18.3202515
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0725784, 16.0718536
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3451691, 14.3457260
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6801910, 16.6824493
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2740631, 17.2753029

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5360942, upper bound: 12.5456227
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5505593, upper bound: 12.5339459
time: 6.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8894615, 13.8899460
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5015182, 8.4986954
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598312, 13.4600983
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0377884, 12.0345497
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6814499, 14.6789169
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1327400, 15.1223183
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6210098, 13.6167221
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1073608, 12.1075039
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0006294, 13.0046005
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9493027, 20.9526749
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0250473, 15.0238647
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6257133, 16.6277351
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7339935, 26.7383423
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8241081, 14.8234596
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2383499, 17.2392426
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4225731, 14.4216576
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5467529, 14.5461502
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6226349, 12.6206703
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9529037, 14.9536133
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3150826, 14.3190346
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2742538, 9.2759743
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8133926, 13.8151855
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3239288, 19.3324051
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2360382, 13.2347260
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9542618, 14.9573479
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5630150, 13.5638390
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6563148, 14.6537704
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2833214, 13.2775459
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6272888, 16.6227684
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1189423, 14.1121521
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9904099, 12.9868698
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0966034, 15.0951691
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9219360, 18.9244919
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3222427, 18.3154678
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0728607, 16.0715675
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3456192, 14.3452892
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6828461, 16.6797905
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2747116, 17.2749138

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5255980, upper bound: 12.5624216
time: 10.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5386175, upper bound: 12.5479763
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8905182, 13.8888893
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4988632, 8.5013542
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4603500, 13.4595833
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0349426, 12.0373955
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6787720, 14.6815987
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1217117, 15.1333427
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6170120, 13.6207237
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1080933, 12.1067715
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0049706, 13.0002556
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535217, 20.9484558
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0236206, 15.0252838
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6275597, 16.6258926
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7390442, 26.7332840
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8242836, 14.8232841
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2403793, 17.2372093
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4214973, 14.4227352
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5457916, 14.5471230
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6202049, 12.6231003
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9528732, 14.9536362
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3191338, 14.3149757
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2756538, 9.2745705
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8151779, 13.8134003
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3319397, 19.3243942
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2343521, 13.2364120
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9571304, 14.9544754
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5638542, 13.5629959
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6535301, 14.6565514
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2768784, 13.2839928
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6212234, 16.6288376
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1109161, 14.1201782
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9853439, 12.9919357
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0931015, 15.0986710
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9233704, 18.9230576
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3129196, 18.3247910
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0693283, 16.0750999
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3452225, 14.3456841
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6790466, 16.6835899
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2750473, 17.2745781

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5505439, upper bound: 12.5356292
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5649890, upper bound: 12.5226076
time: 6.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8902550, 13.8891525
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5014801, 8.4987335
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4601212, 13.4597893
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0379181, 12.0343819
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6802444, 14.6801262
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1314774, 15.1235771
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6210022, 13.6167336
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1080627, 12.1067982
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0015030, 13.0037231
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9511108, 20.9508667
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0248985, 15.0240021
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6252937, 16.6280899
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7353058, 26.7370224
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8249245, 14.8226204
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2402344, 17.2373543
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4218636, 14.4223728
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5460510, 14.5469170
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6216316, 12.6216736
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9519272, 14.9543228
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3158760, 14.3182373
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2740173, 9.2761650
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8134003, 13.8151779
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3239288, 19.3323822
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2351227, 13.2356415
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9542770, 14.9573326
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5630493, 13.5637970
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6555214, 14.6545601
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2821922, 13.2786751
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6250381, 16.6250191
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1172638, 14.1138306
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9878464, 12.9894295
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0936584, 15.0981140
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9209747, 18.9252625
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3177109, 18.3200073
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0696106, 16.0748138
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3456612, 14.3452358
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6817093, 16.6809311
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2754364, 17.2739258

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5365186, upper bound: 12.5475464
time: 11.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5481945, upper bound: 12.5330796
time: 7.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5330797, upper bound: 12.5481944
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5475465, upper bound: 12.5365185
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5226077, upper bound: 12.5649890
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5356293, upper bound: 12.5505438
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5479763, upper bound: 12.5386175
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5624217, upper bound: 12.5255980
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5339459, upper bound: 12.5505593
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5456228, upper bound: 12.5360941
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5360942, upper bound: 12.5456227
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5505593, upper bound: 12.5339459
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5255980, upper bound: 12.5624216
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5386175, upper bound: 12.5479763
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5505439, upper bound: 12.5356292
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5649890, upper bound: 12.5226076
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5365186, upper bound: 12.5475464
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.24
Output dim: 14, lower bound: -12.5481945, upper bound: 12.5330796

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8873215, 13.8896141
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5003586, 8.5023251
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4624481, 13.4636765
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0343399, 12.0383377
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6693649, 14.6661644
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0980263, 15.1019669
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6112747, 13.6147842
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0942535, 12.0984192
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9978981, 12.9969330
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9651260, 20.9694366
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0126190, 15.0117073
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6104698, 16.6036377
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7238388, 26.7248154
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8213615, 14.8253365
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2275887, 17.2329330
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4208221, 14.4199467
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5513535, 14.5497818
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6260834, 12.6245422
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9586411, 14.9546204
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3005676, 14.3015938
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2756119, 9.2734413
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8066025, 13.8060265
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3208160, 19.3142624
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2355843, 13.2344131
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9485168, 14.9462585
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5631371, 13.5624733
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6600838, 14.6599350
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2612038, 13.2613297
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5970230, 16.5901947
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1010094, 14.0999489
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9714279, 12.9631462
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0867157, 15.0762100
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9263954, 18.9199257
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2805405, 18.2689056
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0405884, 16.0266571
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3459702, 14.3464680
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6699371, 16.6682549
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2702827, 17.2744217

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5328043, upper bound: 12.5378249
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5227086, upper bound: 12.5479191
time: 7.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8875237, 13.8894157
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4993439, 8.5033436
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4628601, 13.4632645
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0342636, 12.0384140
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6674728, 14.6680565
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0943375, 15.1056557
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6100235, 13.6160355
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0961304, 12.0965462
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9986267, 12.9962044
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9678955, 20.9666672
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0109406, 15.0133896
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6083412, 16.6057968
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7254868, 26.7231674
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8216438, 14.8250771
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2288094, 17.2317123
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4206848, 14.4201241
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5511703, 14.5499992
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6252899, 12.6253395
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9583817, 14.9549103
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3028107, 14.2993584
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2761459, 9.2729225
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8078079, 13.8048172
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3232346, 19.3118439
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2355499, 13.2344475
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9496613, 14.9451141
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5631943, 13.5624199
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6593132, 14.6607018
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2583618, 13.2641716
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5926666, 16.5945511
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0984306, 14.1025352
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9674149, 12.9671593
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0832443, 15.0796852
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9259224, 18.9204521
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2744293, 18.2750168
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0352173, 16.0320244
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460007, 14.3464546
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6682053, 16.6699867
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716560, 17.2731972

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5472712, upper bound: 12.5261491
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5371760, upper bound: 12.5362433
time: 8.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8870583, 13.8898811
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5029793, 8.4997044
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4622498, 13.4639053
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0373535, 12.0353661
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6708374, 14.6646919
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1077919, 15.0922012
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6152573, 13.6107941
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0942230, 12.0984497
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9944305, 13.0004005
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9627151, 20.9718399
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0139046, 15.0104294
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6083031, 16.6058998
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7201004, 26.7285538
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8220329, 14.8247185
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2274437, 17.2330780
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4212189, 14.4195843
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5515976, 14.5495224
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6275101, 12.6231155
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9579773, 14.9555702
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2973099, 14.3048553
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2740173, 9.2750931
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8048172, 13.8078041
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3128204, 19.3222733
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2363548, 13.2336426
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9456558, 14.9491119
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623360, 13.5632820
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6620750, 14.6579437
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2665176, 13.2560158
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6008377, 16.5863762
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1073647, 14.0936012
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9739304, 12.9606400
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0872726, 15.0756531
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9242439, 18.9223213
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2853241, 18.2641296
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0408707, 16.0263710
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464165, 14.3460484
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6725922, 16.6655960
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2709312, 17.2741852

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5223324, upper bound: 12.5546356
time: 10.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5122368, upper bound: 12.5647144
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8872566, 13.8896790
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5019608, 8.5007229
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4626617, 13.4634933
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0372772, 12.0354385
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6689453, 14.6665878
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1041031, 15.0958900
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6140060, 13.6120453
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0960999, 12.0965729
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9951591, 12.9996719
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9654846, 20.9690704
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0122223, 15.0121117
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6061440, 16.6080551
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7217484, 26.7269058
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8222923, 14.8244362
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2286644, 17.2318573
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4210434, 14.4197216
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5513687, 14.5497360
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6267166, 12.6239128
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9576874, 14.9558487
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2995453, 14.3026199
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2745552, 9.2745552
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8060226, 13.8065987
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3152390, 19.3198547
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2363205, 13.2336769
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9468002, 14.9479675
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623856, 13.5632286
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6613045, 14.6587105
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2636795, 13.2588539
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5964813, 16.5907326
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1047783, 14.0961838
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9699173, 12.9646530
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0838013, 15.0791283
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9237251, 18.9227791
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2792130, 18.2702332
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0355072, 16.0317421
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464279, 14.3460178
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6708603, 16.6673279
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2721596, 17.2728081

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5353539, upper bound: 12.5401743
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5252584, upper bound: 12.5502685
time: 10.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8881149, 13.8888588
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5003204, 8.5023632
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4627609, 13.4633942
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0345078, 12.0382271
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6682053, 14.6673775
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0966911, 15.1032295
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6112595, 13.6147957
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0949554, 12.0977516
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9987717, 12.9960594
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9669342, 20.9676971
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0124779, 15.0118561
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6101418, 16.6040573
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7251511, 26.7235031
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8222008, 14.8245201
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2294731, 17.2311211
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4201050, 14.4206619
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5506058, 14.5504951
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6251030, 12.6255455
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9579773, 14.9555931
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3013687, 14.3007965
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2754250, 9.2736931
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8066025, 13.8060226
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3208313, 19.3142624
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2346687, 13.2353287
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9485321, 14.9462433
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5631752, 13.5624352
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6592903, 14.6607246
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2600746, 13.2624588
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5947723, 16.5924454
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0993385, 14.1016273
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9689102, 12.9657097
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0838928, 15.0791550
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9256172, 18.9208870
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2761841, 18.2734451
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0373383, 16.0299034
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460236, 14.3464241
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6687927, 16.6693954
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2712669, 17.2737007

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5477010, upper bound: 12.5282478
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5376060, upper bound: 12.5383422
time: 12.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8883171, 13.8886566
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4993057, 8.5033817
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4631729, 13.4629784
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0344315, 12.0383034
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6663132, 14.6692696
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0930061, 15.1069145
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6100082, 13.6160469
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0968323, 12.0958786
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9995003, 12.9953308
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9697037, 20.9649277
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0107956, 15.0135345
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6079826, 16.6062202
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7267990, 26.7218552
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8224831, 14.8242607
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2306938, 17.2299042
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4199677, 14.4208393
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5504074, 14.5507126
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6243095, 12.6263428
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9577026, 14.9558830
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3036041, 14.2985611
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2759590, 9.2731590
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8078079, 13.8048134
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3232498, 19.3118439
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2346344, 13.2353630
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9496765, 14.9450989
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5632324, 13.5623856
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6585274, 14.6614914
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2572327, 13.2653008
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5904160, 16.5968018
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0967522, 14.1042099
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9648972, 12.9697227
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0804214, 15.0826263
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9251595, 18.9214134
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2700729, 18.2795563
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0319672, 16.0352745
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460541, 14.3464127
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6670685, 16.6711273
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2726402, 17.2724724

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5621472, upper bound: 12.5152282
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5520679, upper bound: 12.5253227
time: 10.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8878479, 13.8891220
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5029411, 8.4997425
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4625244, 13.4636002
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0374756, 12.0352135
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6696777, 14.6659050
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1064568, 15.0934639
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6152573, 13.6108055
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0949326, 12.0977783
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9953079, 12.9995232
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9645309, 20.9701080
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0137558, 15.0105743
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6078835, 16.6062584
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7214127, 26.7272415
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8228416, 14.8238792
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2293282, 17.2312698
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4205093, 14.4202995
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5508804, 14.5502892
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6265297, 12.6241188
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9570389, 14.9562836
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2981110, 14.3040581
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2737885, 9.2752876
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8048172, 13.8077965
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3128204, 19.3222504
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2354393, 13.2345581
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9456711, 14.9491005
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623703, 13.5632439
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6612816, 14.6587334
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2653885, 13.2571449
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5985870, 16.5886269
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1056862, 14.0952797
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9714203, 12.9632034
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0844498, 15.0785980
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9232979, 18.9230881
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2809677, 18.2686615
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0376282, 16.0296211
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464584, 14.3459949
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6714554, 16.6667328
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716560, 17.2731972

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5336706, upper bound: 12.5401896
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5235753, upper bound: 12.5502840
time: 6.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8880501, 13.8889236
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5019226, 8.5007610
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4629440, 13.4631844
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0374069, 12.0352898
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6677856, 14.6677971
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1027718, 15.0971489
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6139984, 13.6120567
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0968094, 12.0959053
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9960327, 12.9987984
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9673004, 20.9673386
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0120735, 15.0122528
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6057167, 16.6083870
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7230606, 26.7255936
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8231010, 14.8236008
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2305489, 17.2300491
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4203262, 14.4204369
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5506516, 14.5504723
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6257324, 12.6249161
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9567490, 14.9565430
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3003464, 14.3018188
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2743111, 9.2747536
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8060379, 13.8065910
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3152390, 19.3198395
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2354050, 13.2345924
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9468155, 14.9479561
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624237, 13.5631866
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6605110, 14.6595001
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2625504, 13.2599831
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5942307, 16.5929832
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1030998, 14.0978584
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9674072, 12.9672165
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0809784, 15.0820732
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9227638, 18.9235611
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2748566, 18.2747803
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0322571, 16.0349884
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464737, 14.3459644
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6697235, 16.6684647
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2728844, 17.2718201

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5453474, upper bound: 12.5257238
time: 9.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5352520, upper bound: 12.5358188
time: 7.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8878899, 13.8880501
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5005226, 8.5019245
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4627075, 13.4629440
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0347366, 12.0374031
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6678009, 14.6663208
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0971489, 15.1025620
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6115646, 13.6139984
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0948410, 12.0968056
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9982758, 12.9960327
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9659576, 20.9673004
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0122528, 15.0119476
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6083870, 16.6038055
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7245255, 26.7230606
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8221855, 14.8231010
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2287407, 17.2305489
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4204407, 14.4201050
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5504684, 14.5501480
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6249161, 12.6250038
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9565430, 14.9553604
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3006744, 14.3003464
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2747536, 9.2737579
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8065872, 13.8060341
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3198395, 19.3147202
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2345924, 13.2347832
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9479523, 14.9464722
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5631599, 13.5624199
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6595039, 14.6601791
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2599831, 13.2619972
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5929794, 16.5917435
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0978584, 14.1011887
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9672165, 12.9646721
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0820694, 15.0782776
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9235573, 18.9210510
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2747803, 18.2714386
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0349884, 16.0288963
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3459625, 14.3465328
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6684647, 16.6689987
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2703972, 17.2728806

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5358188, upper bound: 12.5352520
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5257239, upper bound: 12.5453474
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8880920, 13.8878479
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4995079, 8.5029430
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4631195, 13.4625282
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0346603, 12.0374794
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6659088, 14.6682129
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0934639, 15.1062546
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6103134, 13.6152534
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0967178, 12.0949287
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9990044, 12.9953041
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9687271, 20.9645309
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0105743, 15.0136299
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6062584, 16.6059685
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7261734, 26.7214127
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8224754, 14.8228416
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2299614, 17.2293282
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4203033, 14.4202824
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5502853, 14.5503654
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6241188, 12.6258011
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9562836, 14.9556503
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3029099, 14.2981110
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2752876, 9.2732353
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8077927, 13.8048286
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3222580, 19.3123016
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2345581, 13.2348137
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9490967, 14.9453278
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5632133, 13.5623703
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6587334, 14.6609459
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2571411, 13.2648392
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5886307, 16.5960999
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0952797, 14.1037750
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9632034, 12.9686852
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0785980, 15.0817528
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230843, 18.9215813
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2686691, 18.2775421
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0296173, 16.0342674
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3459930, 14.3465195
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6667328, 16.6707306
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2717781, 17.2716560

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5502840, upper bound: 12.5235752
time: 15.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5401897, upper bound: 12.5336706
time: 31.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8876266, 13.8883171
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5031471, 8.4993038
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4624939, 13.4631729
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0377426, 12.0344276
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6692734, 14.6648483
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1069145, 15.0927963
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6155624, 13.6100082
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0948105, 12.0968323
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9948082, 12.9995003
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9635544, 20.9697037
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0135345, 15.0106735
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6062202, 16.6060715
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7208023, 26.7267990
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8228569, 14.8224831
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2285957, 17.2306938
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4208374, 14.4197426
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5507126, 14.5498886
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6263428, 12.6235771
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9558792, 14.9563103
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2974167, 14.3036041
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2731590, 9.2754097
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8048172, 13.8078156
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3118439, 19.3227310
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2353630, 13.2340126
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9450989, 14.9493294
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623550, 13.5632324
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6614876, 14.6581879
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2652969, 13.2566833
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5968018, 16.5879288
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1042061, 14.0948410
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9697189, 12.9621658
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0826263, 15.0777245
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9214058, 18.9234467
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2795563, 18.2666550
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0352783, 16.0286140
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464127, 14.3461132
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6711273, 16.6663399
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2710533, 17.2726440

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5253227, upper bound: 12.5520678
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5152283, upper bound: 12.5621471
time: 6.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8878250, 13.8881149
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5021248, 8.5003223
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4629135, 13.4627609
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0376663, 12.0345039
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6673737, 14.6667442
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1032295, 15.0964890
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6143036, 13.6112595
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0966873, 12.0949554
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9955368, 12.9987717
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9663239, 20.9669342
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0118561, 15.0123520
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6040535, 16.6082268
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7224503, 26.7251511
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8231163, 14.8222046
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2298164, 17.2294731
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4206619, 14.4198799
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5504990, 14.5501022
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6255455, 12.6243744
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9555893, 14.9565849
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2996521, 14.3013687
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2736931, 9.2748718
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8060226, 13.8066063
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3142624, 19.3203125
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2353287, 13.2340431
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9462433, 14.9481850
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624046, 13.5631752
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6607246, 14.6589546
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2624588, 13.2595215
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5924454, 16.5922852
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1016273, 14.0974236
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9657059, 12.9661789
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0791550, 15.0811958
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9208870, 18.9239044
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2734451, 18.2727661
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0299072, 16.0339813
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464241, 14.3460827
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6693954, 16.6680717
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2722816, 17.2712669

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5383422, upper bound: 12.5376059
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5282478, upper bound: 12.5477009
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8886833, 13.8872604
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5004845, 8.5019627
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4630127, 13.4626579
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0348969, 12.0372772
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6665878, 14.6675339
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0958900, 15.1038246
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6115494, 13.6140099
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0955429, 12.0960999
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9991531, 12.9951591
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9677734, 20.9654846
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0121117, 15.0120964
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6080513, 16.6042290
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7258530, 26.7217407
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8230247, 14.8222847
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2306252, 17.2286644
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4197235, 14.4208221
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5497360, 14.5508614
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6239128, 12.6260071
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9558487, 14.9563332
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3014755, 14.2995491
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2745590, 9.2740097
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8066025, 13.8060303
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3198547, 19.3147202
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2336769, 13.2356987
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9479675, 14.9464569
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5631981, 13.5623856
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6587105, 14.6609688
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2588539, 13.2631264
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5907364, 16.5939941
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0961800, 14.1028671
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9646530, 12.9672356
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0791245, 15.0812225
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9227791, 18.9220123
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2702332, 18.2759705
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0317383, 16.0321465
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460159, 14.3464890
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6673279, 16.6701355
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2713890, 17.2721596

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5502686, upper bound: 12.5252583
time: 11.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5401744, upper bound: 12.5353539
time: 11.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8888855, 13.8870583
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4994698, 8.5029774
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4634247, 13.4622459
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0348206, 12.0373535
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6646957, 14.6694260
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0922012, 15.1075134
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6102982, 13.6152611
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0974197, 12.0942230
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9998779, 12.9944305
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9705429, 20.9627151
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0104294, 15.0137787
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6058998, 16.6063919
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7275009, 26.7200928
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8233070, 14.8220291
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2318459, 17.2274437
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4195862, 14.4209976
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5495224, 14.5510788
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6231155, 12.6268005
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9555740, 14.9566231
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3037109, 14.2973137
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2750931, 9.2734718
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8078079, 13.8048210
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3222733, 19.3123016
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2336426, 13.2357330
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9491119, 14.9453125
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5632515, 13.5623360
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6579475, 14.6617355
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2560120, 13.2659683
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5863800, 16.5983505
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0936012, 14.1054497
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9606400, 12.9712486
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0756531, 15.0846977
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9223213, 18.9225388
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2641296, 18.2820892
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0263748, 16.0375137
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3460464, 14.3464775
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6655960, 16.6718674
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2727623, 17.2709312

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5647145, upper bound: 12.5122367
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5546356, upper bound: 12.5223323
time: 8.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8884163, 13.8875237
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5031090, 8.4993420
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4627838, 13.4628639
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0378647, 12.0342636
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6680603, 14.6660614
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1056557, 15.0940590
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6155472, 13.6100197
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0955200, 12.0961304
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9956856, 12.9986229
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9653625, 20.9678955
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0133896, 15.0108109
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6057930, 16.6064301
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7221146, 26.7254791
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8236656, 14.8216438
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2304802, 17.2288094
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4201202, 14.4204597
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5499954, 14.5506516
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6253395, 12.6245804
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9549103, 14.9570198
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2982101, 14.3028069
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2729225, 9.2756004
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8048172, 13.8078079
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3118439, 19.3227081
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2344475, 13.2349281
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9451141, 14.9493179
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5623894, 13.5631905
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6607018, 14.6589775
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2641678, 13.2578125
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5945511, 16.5901794
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1025352, 14.0965157
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9671555, 12.9647293
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0796814, 15.0806656
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9204597, 18.9242134
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2750168, 18.2711945
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0320282, 16.0318604
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464546, 14.3460598
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6699829, 16.6674767
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2717781, 17.2716560

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5362433, upper bound: 12.5371759
time: 18.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5261491, upper bound: 12.5472711
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8886185, 13.8873215
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5020866, 8.5003605
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4631958, 13.4624519
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0377884, 12.0343399
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6661682, 14.6679535
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1019669, 15.0977478
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6142960, 13.6112709
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0973969, 12.0942535
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9964104, 12.9978981
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9681320, 20.9651260
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0117073, 15.0124969
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6036339, 16.6085587
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7237625, 26.7238312
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8239250, 14.8213654
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2317009, 17.2275887
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4199448, 14.4205971
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5497818, 14.5508385
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6245422, 12.6253738
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9546204, 14.9572792
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3004532, 14.3005714
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2734451, 9.2750664
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8060226, 13.8065987
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3142624, 19.3202972
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2344131, 13.2349586
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9462585, 14.9481735
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5624428, 13.5631371
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6599312, 14.6597443
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2613297, 13.2606506
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5901947, 16.5945320
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0999489, 14.0990982
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9631424, 12.9687424
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0762100, 15.0841408
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9199257, 18.9246864
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2689056, 18.2773056
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0266571, 16.0372314
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3464661, 14.3460293
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6682587, 16.6692085
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2729988, 17.2702789

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5479192, upper bound: 12.5227086
time: 10.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5378250, upper bound: 12.5328043
time: 11.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5328043, upper bound: 12.5378249
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5227086, upper bound: 12.5479191
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5472712, upper bound: 12.5261491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5371760, upper bound: 12.5362433
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5223324, upper bound: 12.5546356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5122368, upper bound: 12.5647144
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5353539, upper bound: 12.5401743
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5252584, upper bound: 12.5502685
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5477010, upper bound: 12.5282478
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5376060, upper bound: 12.5383422
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5621472, upper bound: 12.5152282
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5520679, upper bound: 12.5253227
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5336706, upper bound: 12.5401896
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5235753, upper bound: 12.5502840
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5453474, upper bound: 12.5257238
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5352520, upper bound: 12.5358188
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5358188, upper bound: 12.5352520
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5257239, upper bound: 12.5453474
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5502840, upper bound: 12.5235752
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5401897, upper bound: 12.5336706
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5253227, upper bound: 12.5520678
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5152283, upper bound: 12.5621471
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5383422, upper bound: 12.5376059
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5282478, upper bound: 12.5477009
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5502686, upper bound: 12.5252583
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5401744, upper bound: 12.5353539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5647145, upper bound: 12.5122367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5546356, upper bound: 12.5223323
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5362433, upper bound: 12.5371759
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5261491, upper bound: 12.5472711
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5479192, upper bound: 12.5227086
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 14, lower bound: -12.5378250, upper bound: 12.5328043

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8683090, 13.8679390
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4950485, 8.4964523
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4597168, 13.4605980
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0297699, 12.0333176
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6782227, 14.6764526
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1034889, 15.1074829
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6053543, 13.6083450
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0930786, 12.0972195
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9693031, 12.9643364
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9555359, 20.9585495
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0288124, 15.0255890
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5898743, 16.5802994
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7084274, 26.7072906
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7935791, 14.7936554
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2214050, 17.2244949
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4190941, 14.4205513
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5366669, 14.5368996
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6141129, 12.6140442
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9383011, 14.9367714
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2929573, 14.2948303
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2475014, 9.2487869
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7907562, 13.7921295
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3020592, 19.2976151
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2121658, 13.2141037
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9349289, 14.9342155
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5541725, 13.5543823
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6397018, 14.6420441
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2677460, 13.2680740
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5967484, 16.5899429
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0775642, 14.0793877
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9529343, 12.9468269
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0651093, 15.0568924
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9280319, 18.9215851
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2429657, 18.2359543
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0329628, 16.0193977
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593140, 14.3585815
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6690826, 16.6674652
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2880821, 17.2900620

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4903810, upper bound: 12.5376428
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5326215, upper bound: 12.4954127
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8656464, 13.8706017
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4944878, 8.4970169
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4593735, 13.4609451
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0293198, 12.0337677
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6796570, 14.6750183
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1035385, 15.1074295
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6048279, 13.6088676
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0930557, 12.0972462
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9652977, 12.9683418
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9542465, 20.9598465
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0265045, 15.0279007
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5871353, 16.5830383
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7063065, 26.7094116
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7896805, 14.7975502
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2191467, 17.2267494
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4214287, 14.4182167
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5384674, 14.5350952
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6155815, 12.6125717
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9407883, 14.9342842
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2938118, 14.2939796
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2509537, 9.2453308
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7927017, 13.7901840
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3041649, 19.2955017
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2152748, 13.2109947
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9364777, 14.9326668
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5550461, 13.5535049
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6421890, 14.6395531
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2679443, 13.2678719
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5967636, 16.5899277
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0804482, 14.0765038
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9551086, 12.9446564
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0673981, 15.0546036
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9280548, 18.9215622
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2475891, 18.2313385
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0333290, 16.0190315
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3580818, 14.3598118
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6691437, 16.6674004
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2859230, 17.2922249

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4802967, upper bound: 12.5477363
time: 14.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5225265, upper bound: 12.5054960
time: 8.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8685112, 13.8677368
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4940300, 8.4974670
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4601288, 13.4601860
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0296936, 12.0333939
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6763229, 14.6783447
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0998001, 15.1111717
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6041031, 13.6095963
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0949554, 12.0953465
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9700317, 12.9636078
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9583054, 20.9557800
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0271301, 15.0272713
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5877457, 16.5824585
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7100754, 26.7056427
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7938614, 14.7933960
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2226257, 17.2232742
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4189568, 14.4207287
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5364761, 14.5371170
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6133194, 12.6148376
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9380417, 14.9370613
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2951927, 14.2925949
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2480354, 9.2482681
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7919617, 13.7909241
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3044777, 19.2951965
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2121315, 13.2141342
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9360733, 14.9330711
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542259, 13.5543289
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6389313, 14.6428146
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2649078, 13.2709122
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5923920, 16.5942955
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0749855, 14.0819702
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9489212, 12.9508400
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0616379, 15.0603638
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9275589, 18.9221115
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2368622, 18.2420654
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0275993, 16.0247688
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593445, 14.3585682
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6673584, 16.6691971
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2894630, 17.2888412

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5048481, upper bound: 12.5259669
time: 16.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5470884, upper bound: 12.4837368
time: 11.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8658447, 13.8703995
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4934692, 8.4980316
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4597855, 13.4605293
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0292511, 12.0338440
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6777649, 14.6769104
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0998535, 15.1111183
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6035767, 13.6101227
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0949249, 12.0953732
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9660263, 12.9676132
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9570084, 20.9570770
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0248222, 15.0295792
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5850067, 16.5852013
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7079544, 26.7077637
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7899704, 14.7972908
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2203674, 17.2255287
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4212914, 14.4183922
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5382843, 14.5353127
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6147881, 12.6133652
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9405289, 14.9345741
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2960472, 14.2917442
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2514915, 9.2448120
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7939072, 13.7889748
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3065834, 19.2930832
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2152405, 13.2110291
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9376221, 14.9315224
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5551033, 13.5534554
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6414261, 14.6403198
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2651062, 13.2707138
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5924072, 16.5942802
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0778618, 14.0790863
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9510956, 12.9486694
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0639267, 15.0580750
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9275818, 18.9220886
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2414703, 18.2374496
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0279655, 16.0244026
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3581123, 14.3597984
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6674194, 16.6691322
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2872963, 17.2910004

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4947640, upper bound: 12.5360605
time: 15.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5369939, upper bound: 12.4938201
time: 11.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8680420, 13.8682022
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4976692, 8.4938316
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4595108, 13.4608269
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0327835, 12.0303459
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6796951, 14.6749802
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1132545, 15.0977173
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6093445, 13.6043549
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0930481, 12.0972500
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9658356, 12.9678001
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9531250, 20.9609604
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0300941, 15.0243149
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5877075, 16.5825653
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7046890, 26.7110367
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7942429, 14.7930374
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2212601, 17.2246399
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4194908, 14.4201889
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5369034, 14.5366402
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6155396, 12.6126175
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9376450, 14.9377213
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2896919, 14.2980919
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2459068, 9.2504349
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7889786, 13.7939110
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2940636, 19.3056183
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2129364, 13.2133331
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9320679, 14.9370728
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5533676, 13.5551910
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6416855, 14.6400528
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2730637, 13.2627563
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6005707, 16.5861244
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0839119, 14.0730362
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9554443, 12.9443207
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0656662, 15.0563354
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9258881, 18.9239807
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2477570, 18.2311783
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0332451, 16.0191154
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3597603, 14.3581619
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6717377, 16.6648064
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2887306, 17.2898254

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4799087, upper bound: 12.5544536
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5221496, upper bound: 12.5122261
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8653793, 13.8708687
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4971046, 8.4943962
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591675, 13.4611740
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0323334, 12.0307961
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6811295, 14.6735458
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1133080, 15.0976639
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6088257, 13.6048775
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0930252, 12.0972729
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9618301, 12.9718056
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9518356, 20.9622498
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0277863, 15.0266228
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5849609, 16.5853043
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7025833, 26.7131500
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7903519, 14.7969322
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2190018, 17.2268944
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4218254, 14.4178543
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5387115, 14.5348358
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6170082, 12.6111450
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9401321, 14.9352341
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2905464, 14.2972374
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2493591, 9.2469826
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7909241, 13.7919617
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2961769, 19.3035126
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2160454, 13.2102242
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9336166, 14.9355240
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542450, 13.5543137
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6441803, 14.6375580
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2732620, 13.2625580
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6005859, 16.5861053
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0867958, 14.0701561
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9576111, 12.9421501
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0679550, 15.0540466
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9259109, 18.9239578
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2523651, 18.2265549
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0336113, 16.0187492
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585281, 14.3593941
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6717987, 16.6647415
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2865715, 17.2919884

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4698247, upper bound: 12.5645319
time: 15.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5120547, upper bound: 12.5223091
time: 7.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8682442, 13.8680000
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4966507, 8.4948502
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4599304, 13.4604149
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0327072, 12.0304222
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6777954, 14.6768723
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1095657, 15.1014061
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6080933, 13.6056061
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0949249, 12.0953732
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9665642, 12.9670753
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9558945, 20.9581909
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0284119, 15.0259933
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5855408, 16.5847168
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7063370, 26.7093887
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7945023, 14.7927589
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2224808, 17.2234192
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4193153, 14.4203281
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5366898, 14.5368538
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6147423, 12.6134109
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9373550, 14.9379959
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2919273, 14.2958565
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2464409, 9.2499008
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7901840, 13.7927017
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2964821, 19.3032074
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2129021, 13.2133636
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9332123, 14.9359283
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5534210, 13.5551376
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6409225, 14.6408234
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2702217, 13.2655945
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5962143, 16.5904770
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0813332, 14.0756187
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9514313, 12.9483337
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0621948, 15.0598068
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9253616, 18.9244385
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2416382, 18.2372818
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0278816, 16.0244827
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3597755, 14.3581314
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6700134, 16.6665382
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2899666, 17.2884521

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4929306, upper bound: 12.5399922
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5351711, upper bound: 12.4977621
time: 7.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8655815, 13.8706665
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4960899, 8.4954147
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4595795, 13.4607582
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0322571, 12.0308723
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6792374, 14.6754379
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1096191, 15.1013527
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6075745, 13.6061287
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0949020, 12.0953999
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9625587, 12.9710808
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9546051, 20.9594879
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0261040, 15.0283051
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5828018, 16.5874596
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7042313, 26.7115021
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7906113, 14.7966499
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2202225, 17.2256737
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4216499, 14.4179916
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5384903, 14.5350494
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6162148, 12.6119385
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9398422, 14.9355087
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2927818, 14.2950020
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2498970, 9.2464447
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7921295, 13.7907524
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2985954, 19.3010941
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2160110, 13.2102585
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9347610, 14.9343796
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542946, 13.5542603
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6434174, 14.6383286
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2704201, 13.2653961
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5962296, 16.5904617
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0842171, 14.0727386
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9535980, 12.9461632
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0644836, 15.0575180
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9253845, 18.9244156
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2462616, 18.2326660
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0282478, 16.0241203
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585396, 14.3593636
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6700745, 16.6664772
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2877998, 17.2906113

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4828465, upper bound: 12.5500857
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5250763, upper bound: 12.5078453
time: 25.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8690987, 13.8671799
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4950104, 8.4964905
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600220, 13.4603157
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0299377, 12.0332069
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6770554, 14.6776657
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1021538, 15.1087418
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6053467, 13.6083565
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0937805, 12.0965519
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9701805, 12.9634590
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9573517, 20.9568100
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0286713, 15.0257378
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5895386, 16.5807190
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7097549, 26.7059860
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7944183, 14.7928391
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2232895, 17.2226830
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4183769, 14.4212685
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5359268, 14.5376129
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6131325, 12.6150436
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9376373, 14.9377441
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2937508, 14.2940331
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2473145, 9.2490387
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7907639, 13.7921257
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3020744, 19.2976151
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2112465, 13.2150192
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9349365, 14.9342003
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542107, 13.5543442
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6389084, 14.6428337
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2666168, 13.2692032
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5944977, 16.5921898
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0758858, 14.0810623
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9504242, 12.9493904
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0622864, 15.0598335
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9272537, 18.9225426
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2386169, 18.2405014
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0297203, 16.0226479
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593674, 14.3585377
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6679382, 16.6686020
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2890739, 17.2893448

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5052777, upper bound: 12.5280657
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5475182, upper bound: 12.4858359
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8664360, 13.8698425
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4944496, 8.4970512
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4596786, 13.4606590
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0294876, 12.0336571
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6784973, 14.6762276
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1022072, 15.1086922
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6048203, 13.6088791
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0937576, 12.0965786
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9661751, 12.9674644
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9560547, 20.9581070
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0263634, 15.0280457
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5867996, 16.5834618
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7076340, 26.7080994
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7905197, 14.7967339
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2210312, 17.2249413
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4207115, 14.4189339
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5377350, 14.5358086
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6146011, 12.6135750
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9401245, 14.9352570
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2946053, 14.2931824
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2507668, 9.2455826
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7927094, 13.7901764
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3041801, 19.2955017
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2143555, 13.2119102
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9364853, 14.9326553
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5550842, 13.5534706
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6414032, 14.6403427
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2668152, 13.2690010
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5945129, 16.5921745
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0787697, 14.0781822
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9525909, 12.9472198
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0645752, 15.0575447
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9272766, 18.9225197
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2432404, 18.2358780
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0300865, 16.0222816
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3581352, 14.3597679
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6680145, 16.6685410
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2869072, 17.2915039

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4951938, upper bound: 12.5381594
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5374239, upper bound: 12.4959194
time: 8.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8693008, 13.8669777
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4939919, 8.4975052
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4604416, 13.4598999
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0298615, 12.0332832
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6751633, 14.6795578
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0984650, 15.1124306
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6040955, 13.6096115
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0956573, 12.0946789
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9709091, 12.9627304
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9601135, 20.9540482
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0269890, 15.0274200
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5873871, 16.5828819
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7114029, 26.7043381
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7947006, 14.7925797
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2245102, 17.2214622
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4182396, 14.4214439
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5357132, 14.5378342
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6123352, 12.6158409
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9373627, 14.9380302
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2959862, 14.2917976
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2478485, 9.2485008
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7919693, 13.7909164
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3044930, 19.2951965
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2112160, 13.2150536
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9360809, 14.9330559
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542641, 13.5542946
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6381378, 14.6436005
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2637787, 13.2720413
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5901489, 16.5965462
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0733070, 14.0836449
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9464111, 12.9534035
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0588150, 15.0633087
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9267960, 18.9230728
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2325134, 18.2466049
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0243492, 16.0280151
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593979, 14.3585243
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6662140, 16.6703339
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2904472, 17.2881126

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5197422, upper bound: 12.5150461
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5619646, upper bound: 12.4728162
time: 8.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8666382, 13.8696442
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4934311, 8.4980698
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600983, 13.4602470
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0294113, 12.0337334
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6765976, 14.6781235
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0985184, 15.1123772
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6035690, 13.6101341
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0956345, 12.0947018
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9668999, 12.9667358
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9588242, 20.9553375
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0246811, 15.0297279
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5846481, 16.5856247
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7092819, 26.7064514
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7908020, 14.7964745
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2222519, 17.2237206
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4205742, 14.4191093
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5375214, 14.5360260
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6138077, 12.6143684
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9398499, 14.9355469
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2968407, 14.2909470
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2513008, 9.2450485
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7939148, 13.7889709
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3065987, 19.2930908
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2143250, 13.2119446
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9376297, 14.9315109
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5551414, 13.5534210
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6406326, 14.6411095
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2639771, 13.2718430
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5901642, 16.5965309
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0761909, 14.0807648
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9485779, 12.9512329
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0611038, 15.0610199
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9268188, 18.9230499
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2371216, 18.2419815
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0247154, 16.0276527
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3581657, 14.3597565
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6662750, 16.6702728
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2882881, 17.2902756

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5096588, upper bound: 12.5251399
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5518859, upper bound: 12.4828996
time: 8.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8688354, 13.8674469
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4976311, 8.4938698
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4597931, 13.4605217
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0329132, 12.0301971
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6785278, 14.6761932
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1119194, 15.0989761
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6093369, 13.6043663
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0937576, 12.0965786
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9667130, 12.9669266
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9549408, 20.9592209
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0299454, 15.0244560
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5872803, 16.5829201
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7060013, 26.7097244
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7950516, 14.7921982
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2231445, 17.2228279
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4187737, 14.4209061
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5361862, 14.5374069
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6145592, 12.6136208
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9367065, 14.9384308
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2904930, 14.2972946
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2456779, 9.2506332
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7889786, 13.7939034
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2940636, 19.3056030
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2120209, 13.2142487
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9320831, 14.9370575
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5534058, 13.5551529
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6408997, 14.6408424
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2719345, 13.2638855
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5983200, 16.5883713
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0822411, 14.0747147
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9529266, 12.9468842
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0628433, 15.0592804
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9249268, 18.9247437
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2433929, 18.2357101
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0300026, 16.0223618
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3598022, 14.3581085
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6706085, 16.6659431
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2894630, 17.2888374

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4912470, upper bound: 12.5400075
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5334878, upper bound: 12.4977778
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8661728, 13.8701096
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4970665, 8.4944344
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594498, 13.4608650
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0324554, 12.0306435
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6799698, 14.6747589
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1119728, 15.0989265
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6088104, 13.6048889
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0937271, 12.0966053
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9627075, 12.9709320
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9536438, 20.9605179
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0276375, 15.0267639
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5845413, 16.5856628
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7038956, 26.7118378
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7911606, 14.7960930
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2208939, 17.2250862
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4211159, 14.4185715
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5379944, 14.5356026
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6160278, 12.6121483
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9391937, 14.9359436
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2913475, 14.2964401
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2491302, 9.2471771
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7909317, 13.7919540
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2961693, 19.3034973
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2151299, 13.2111435
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9336319, 14.9355087
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542793, 13.5542755
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6433945, 14.6383514
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2721329, 13.2636871
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5983353, 16.5883560
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0851173, 14.0718307
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9551010, 12.9447136
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0651321, 15.0569916
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9249496, 18.9247208
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2480164, 18.2311020
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0303688, 16.0219994
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585701, 14.3593407
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6706696, 16.6658821
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2872963, 17.2910004

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4811630, upper bound: 12.5501012
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5233931, upper bound: 12.5078612
time: 7.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8690376, 13.8672447
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4966125, 8.4948883
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4602051, 13.4601059
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0328369, 12.0302696
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6766357, 14.6780853
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1082306, 15.1026649
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6080780, 13.6056175
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0956268, 12.0947056
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9674416, 12.9661980
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9577103, 20.9564514
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0282669, 15.0261383
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5851212, 16.5850487
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7076492, 26.7080765
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7953110, 14.7919197
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2243652, 17.2216110
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4185982, 14.4210434
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5359726, 14.5375900
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6137619, 12.6144142
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9364166, 14.9386902
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2927284, 14.2950592
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2461967, 9.2500954
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7901917, 13.7926941
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2964821, 19.3031921
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2119865, 13.2142792
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9332275, 14.9359131
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5534554, 13.5550995
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6401291, 14.6416092
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2690926, 13.2667236
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5939636, 16.5927277
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0796547, 14.0772972
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9489136, 12.9508972
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0593719, 15.0627518
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9244003, 18.9252167
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2372894, 18.2418289
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0246315, 16.0277328
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3598137, 14.3580780
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6688690, 16.6676750
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2906837, 17.2874603

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5029238, upper bound: 12.5255418
time: 15.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5451646, upper bound: 12.4833121
time: 27.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8663712, 13.8699074
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4960518, 8.4954491
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598618, 13.4604530
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0323792, 12.0307198
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6780701, 14.6766510
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1082840, 15.1026115
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6075592, 13.6061401
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0956039, 12.0947323
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9634323, 12.9702034
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9564133, 20.9577484
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0259552, 15.0284462
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5823822, 16.5877914
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7055435, 26.7101898
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7914200, 14.7958145
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2221069, 17.2238655
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4209328, 14.4187088
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5377808, 14.5357857
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6152344, 12.6129417
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9389038, 14.9362030
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2935829, 14.2942047
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2496529, 9.2466431
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7921371, 13.7907448
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2985954, 19.3010788
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2150955, 13.2111740
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9347763, 14.9343643
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5543289, 13.5542221
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6426239, 14.6391182
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2692909, 13.2665253
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5939789, 16.5927124
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0825386, 14.0744133
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9510880, 12.9487267
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0616608, 15.0604630
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9244232, 18.9251938
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2418976, 18.2372055
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0249977, 16.0273666
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585854, 14.3593102
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6689301, 16.6676140
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2885246, 17.2896233

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4928397, upper bound: 12.5356360
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5350699, upper bound: 12.4933958
time: 10.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8688812, 13.8663712
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4952164, 8.4960480
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4599609, 13.4598656
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0301666, 12.0323830
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6766510, 14.6766014
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1026115, 15.1080818
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6056519, 13.6075592
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0936737, 12.0956020
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9696770, 12.9634361
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9563675, 20.9564133
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0284462, 15.0258255
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5877914, 16.5804749
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7091141, 26.7055435
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7943954, 14.7914200
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2225494, 17.2221107
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4187050, 14.4207115
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5357895, 14.5372658
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6129417, 12.6145020
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9362030, 14.9375114
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2930565, 14.2935829
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2466393, 9.2491074
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7907486, 13.7921410
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3010750, 19.2980804
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2111740, 13.2144661
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9343643, 14.9344254
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5541878, 13.5543289
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6391144, 14.6422920
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2665253, 13.2687416
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5927124, 16.5914879
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0744133, 14.0806274
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9487228, 12.9483566
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0604630, 15.0589638
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9251938, 18.9227142
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2372131, 18.2384872
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0273705, 16.0216331
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593102, 14.3586445
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6676178, 16.6682129
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2882195, 17.2885246

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4933958, upper bound: 12.5350698
time: 9.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5356361, upper bound: 12.4928397
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8662186, 13.8690376
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4946518, 8.4966125
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4596252, 13.4602089
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0297165, 12.0328331
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6780853, 14.6751633
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1026649, 15.1080284
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6051254, 13.6080818
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0936508, 12.0956287
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9656677, 12.9674416
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9550705, 20.9577103
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0261383, 15.0281372
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5850525, 16.5832138
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7070084, 26.7076569
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7905045, 14.7953148
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2202911, 17.2243652
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4210396, 14.4183769
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5375900, 14.5354614
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6144142, 12.6130333
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9386902, 14.9350243
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2939110, 14.2927284
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2500954, 9.2456551
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7926941, 13.7901917
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3031883, 19.2959671
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2142830, 13.2113609
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9359131, 14.9328804
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5550652, 13.5534554
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6416092, 14.6398010
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2667236, 13.2685394
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5927277, 16.5914726
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0772972, 14.0777473
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9508972, 12.9461861
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0627518, 15.0566750
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9252167, 18.9226913
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2418213, 18.2338715
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0277367, 16.0212708
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3580780, 14.3598747
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6676788, 16.6681480
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2860603, 17.2906837

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4833122, upper bound: 12.5451646
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5255418, upper bound: 12.5029237
time: 11.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8690834, 13.8661728
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4941978, 8.4970665
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4603806, 13.4594498
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0300903, 12.0324593
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6747589, 14.6784935
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0989227, 15.1117706
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6044006, 13.6088104
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0955505, 12.0937290
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9704018, 12.9627075
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9591370, 20.9536438
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0267639, 15.0275078
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5856628, 16.5826340
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7107620, 26.7038956
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7946854, 14.7911606
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2237701, 17.2208900
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4185677, 14.4208870
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5355988, 14.5374832
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6121483, 12.6152992
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9359436, 14.9378014
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2952919, 14.2913475
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2471771, 9.2485886
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7919540, 13.7909317
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3034935, 19.2956619
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2111397, 13.2145004
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9355087, 14.9332809
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542412, 13.5542793
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6383514, 14.6430588
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2636871, 13.2715797
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5883560, 16.5958443
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0718269, 14.0832138
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9447098, 12.9523697
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0569916, 15.0624390
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9247208, 18.9232407
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2310944, 18.2445984
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0219994, 16.0270042
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593407, 14.3586311
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6658783, 16.6699448
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2895927, 17.2873001

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5078612, upper bound: 12.5233931
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5501012, upper bound: 12.4811630
time: 8.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8664169, 13.8688354
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4936371, 8.4976311
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600372, 13.4597969
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0296402, 12.0329094
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6761932, 14.6770592
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0989761, 15.1117172
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6038742, 13.6093369
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0955200, 12.0937557
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9663963, 12.9667130
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9578400, 20.9549408
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0244560, 15.0298157
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5829163, 16.5853767
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7086563, 26.7060089
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7907867, 14.7950554
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2215118, 17.2231445
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4209023, 14.4185524
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5374069, 14.5356789
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6136208, 12.6138268
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9384308, 14.9353142
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2961464, 14.2904930
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2506294, 9.2451324
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7938995, 13.7889862
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3056068, 19.2935486
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2142487, 13.2113914
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9370575, 14.9317360
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5551186, 13.5534058
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6408463, 14.6405678
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2638855, 13.2713814
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5883713, 16.5958290
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0747108, 14.0803299
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9468842, 12.9501991
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0592804, 15.0601501
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9247437, 18.9232178
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2357178, 18.2399826
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0223656, 16.0266380
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3581085, 14.3598633
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6659393, 16.6698799
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2874336, 17.2894592

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4977779, upper bound: 12.5334878
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5400076, upper bound: 12.4912469
time: 11.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8686142, 13.8666382
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4978333, 8.4934311
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4597626, 13.4600945
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0331802, 12.0294113
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6781235, 14.6751289
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1123772, 15.0983162
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6096420, 13.6035690
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0936432, 12.0956326
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9662094, 12.9668999
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9539566, 20.9588242
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0297279, 15.0245514
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5856247, 16.5827408
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7053757, 26.7092819
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7950668, 14.7908020
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2224045, 17.2222557
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4191093, 14.4203491
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5360260, 14.5370064
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6143684, 12.6130791
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9355469, 14.9384613
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2897987, 14.2968445
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2450485, 9.2507591
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7889709, 13.7939186
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2930870, 19.3060913
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2119446, 13.2136993
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9315109, 14.9372826
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5533867, 13.5551414
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6411057, 14.6403008
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2718430, 13.2634239
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5965271, 16.5876732
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0807610, 14.0742798
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9512329, 12.9458504
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0610199, 15.0584068
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230499, 18.9251060
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2419891, 18.2337036
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0276527, 16.0213509
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3597565, 14.3582249
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6702728, 16.6655502
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2888680, 17.2882843

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4828996, upper bound: 12.5518859
time: 20.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5251399, upper bound: 12.5096586
time: 6.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8659515, 13.8693008
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4972725, 8.4939919
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4594116, 13.4604378
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0327301, 12.0298615
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6795578, 14.6736946
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1124306, 15.0982628
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6091156, 13.6040916
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0936203, 12.0956593
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9622040, 12.9709053
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9526596, 20.9601135
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0274200, 15.0268593
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5828857, 16.5854797
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7032547, 26.7113953
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7911682, 14.7946968
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2201462, 17.2245102
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4214439, 14.4180145
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5378342, 14.5352020
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6158409, 12.6116066
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9380341, 14.9359741
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2906532, 14.2959900
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2485008, 9.2473068
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7909164, 13.7919731
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2951927, 19.3039780
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2150536, 13.2105904
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9330597, 14.9357376
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542641, 13.5542641
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6436005, 14.6378098
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2720413, 13.2632256
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5965424, 16.5876579
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0836449, 14.0713959
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9533997, 12.9436798
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0633087, 15.0561180
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9230728, 18.9250832
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2465973, 18.2290878
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0280190, 16.0209846
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585243, 14.3594570
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6703339, 16.6654892
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2867088, 17.2904472

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4728162, upper bound: 12.5619645
time: 14.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5150461, upper bound: 12.5197422
time: 10.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8688164, 13.8664360
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4968185, 8.4944458
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4601746, 13.4596825
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0331039, 12.0294876
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6762314, 14.6770210
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1086884, 15.1020050
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6083832, 13.6048203
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0955200, 12.0937557
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9669342, 12.9661751
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9567261, 20.9560547
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0280457, 15.0262299
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5834656, 16.5848923
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7070236, 26.7076340
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7953262, 14.7905235
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2236252, 17.2210350
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4189339, 14.4204865
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5358124, 14.5372200
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6135750, 12.6138725
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9352570, 14.9387398
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2920341, 14.2946091
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2455826, 9.2502213
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7901764, 13.7927094
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2955055, 19.3036728
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2119102, 13.2137299
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9326553, 14.9361382
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5534363, 13.5550842
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6403427, 14.6410675
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2690010, 13.2662659
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5921783, 16.5920258
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0781822, 14.0768623
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9472198, 12.9498634
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0575485, 15.0618820
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9225235, 18.9255676
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2358856, 18.2398148
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0222816, 16.0267220
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3597679, 14.3581944
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6685486, 16.6672821
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2901039, 17.2869110

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4959195, upper bound: 12.5374238
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5381594, upper bound: 12.4951938
time: 9.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8661537, 13.8690987
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4962540, 8.4950104
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4598236, 13.4600258
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0326538, 12.0299377
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6776657, 14.6755867
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1087418, 15.1019516
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6078644, 13.6053429
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0954971, 12.0937824
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9629288, 12.9701805
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9554291, 20.9573517
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0257378, 15.0285416
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5807190, 16.5876350
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7049026, 26.7097473
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7914276, 14.7944183
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2213669, 17.2232933
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4212685, 14.4181519
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5376129, 14.5354156
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6150436, 12.6124001
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9377441, 14.9362526
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2928886, 14.2937546
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2490387, 9.2467690
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7921295, 13.7907639
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2976112, 19.3015594
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2150192, 13.2106209
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9342041, 14.9345932
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5543137, 13.5542107
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6428375, 14.6385765
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2692032, 13.2660637
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5921936, 16.5920105
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0810661, 14.0739822
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9493866, 12.9476929
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0598373, 15.0595932
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9225464, 18.9255447
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2404938, 18.2351990
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0226479, 16.0263557
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585396, 14.3594265
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6686096, 16.6672211
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2879372, 17.2890701

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4858360, upper bound: 12.5475181
time: 10.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5280657, upper bound: 12.5052776
time: 10.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8696709, 13.8655815
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4951782, 8.4960861
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4602737, 13.4595795
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0303345, 12.0322571
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6754379, 14.6778107
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1013527, 15.1093445
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6056366, 13.6075706
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0943756, 12.0949001
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9705505, 12.9625587
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9581757, 20.9546051
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0283051, 15.0259743
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5874634, 16.5808945
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7104263, 26.7042236
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7952347, 14.7906075
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2244339, 17.2202225
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4179878, 14.4214287
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5350494, 14.5379791
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6119385, 12.6155090
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9355087, 14.9384842
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2938576, 14.2927856
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2464485, 9.2493591
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7907562, 13.7921371
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3010979, 19.2980804
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2102585, 13.2153854
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9343796, 14.9344139
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542259, 13.5542946
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6383286, 14.6430817
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2653961, 13.2698708
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5904617, 16.5937386
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0727348, 14.0823059
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9461594, 12.9509201
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0575180, 15.0619087
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9244156, 18.9236717
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2326660, 18.2430267
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0241203, 16.0248833
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593636, 14.3586006
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6664734, 16.6693497
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2892036, 17.2878036

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5078454, upper bound: 12.5250763
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5500858, upper bound: 12.4828465
time: 6.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8670082, 13.8682442
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4946136, 8.4966507
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4599304, 13.4599266
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0298767, 12.0327072
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6768723, 14.6763763
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1014061, 15.1092911
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6051178, 13.6080933
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0943527, 12.0949268
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9665451, 12.9665642
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9568787, 20.9558945
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0259933, 15.0282822
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5847168, 16.5836372
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7083206, 26.7063370
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7913437, 14.7945023
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2221756, 17.2224808
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4203300, 14.4190922
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5368576, 14.5361748
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6134109, 12.6140366
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9379959, 14.9359970
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2947121, 14.2919312
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2499008, 9.2459068
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7927017, 13.7901878
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3032036, 19.2959747
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2133675, 13.2122765
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9359283, 14.9328651
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5551033, 13.5534210
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6408234, 14.6405869
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2655945, 13.2696686
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5904770, 16.5937233
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0756187, 14.0794220
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9483337, 12.9487495
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0598068, 15.0596199
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9244385, 18.9236488
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2372894, 18.2384109
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0244865, 16.0245171
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3581314, 14.3598328
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6665344, 16.6692886
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2870445, 17.2899628

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4977621, upper bound: 12.5351711
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5399922, upper bound: 12.4929305
time: 13.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8698730, 13.8653793
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4941597, 8.4971046
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4606934, 13.4591637
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0302582, 12.0323334
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6735458, 14.6797028
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0976639, 15.1130333
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6043854, 13.6088257
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0962524, 12.0930233
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9712791, 12.9618301
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9609451, 20.9518356
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0266228, 15.0276566
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5853043, 16.5830574
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7120743, 26.7025757
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7955170, 14.7903481
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2256546, 17.2190018
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4178505, 14.4216042
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5348358, 14.5381966
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6111450, 12.6163025
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9352341, 14.9387741
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2960930, 14.2905502
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2469826, 9.2488251
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7919617, 13.7909279
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3035164, 19.2956619
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2102242, 13.2154160
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9355240, 14.9332695
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542831, 13.5542450
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6375580, 14.6438484
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2625580, 13.2727089
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5861053, 16.5980949
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0701561, 14.0848885
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9421463, 12.9549332
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0540466, 15.0653801
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9239578, 18.9241982
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2265625, 18.2491379
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0187492, 16.0302544
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3593941, 14.3585892
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6647491, 16.6710815
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2905846, 17.2865753

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5223092, upper bound: 12.5120546
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5645319, upper bound: 12.4698246
time: 7.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8672104, 13.8680420
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4935989, 8.4976692
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4603424, 13.4595108
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0298080, 12.0327835
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6749802, 14.6782684
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.0977173, 15.1129799
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6038666, 13.6093483
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0962296, 12.0930500
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9672737, 12.9658394
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9596481, 20.9531250
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0243149, 15.0299644
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5825653, 16.5858002
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7099686, 26.7046890
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7916183, 14.7942429
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2233963, 17.2212601
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4201927, 14.4192696
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5366440, 14.5363922
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6126175, 12.6148300
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9377213, 14.9362869
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2969475, 14.2896957
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2504349, 9.2453690
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7939072, 13.7889786
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3056221, 19.2935486
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2133331, 13.2123070
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9370728, 14.9317207
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5551567, 13.5533676
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6400528, 14.6413574
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2627563, 13.2725105
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5861206, 16.5980797
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0730400, 14.0820084
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9443207, 12.9527626
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0563354, 15.0630913
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9239807, 18.9241753
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2311707, 18.2445145
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0191154, 16.0298882
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3581619, 14.3598213
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6648102, 16.6710205
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2884254, 17.2887344

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5122261, upper bound: 12.5221495
time: 15.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5544536, upper bound: 12.4799087
time: 7.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8694077, 13.8658447
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4977989, 8.4934692
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4600449, 13.4597855
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0333099, 12.0292473
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6769104, 14.6763382
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1111183, 15.0995789
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6096268, 13.6035805
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0943527, 12.0949268
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9670830, 12.9660263
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9557648, 20.9570084
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0295792, 15.0246925
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5852051, 16.5830994
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7067032, 26.7079620
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7958755, 14.7899666
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2242889, 17.2203674
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4183922, 14.4210663
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5353165, 14.5377693
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6133652, 12.6140823
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9345703, 14.9391708
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2905998, 14.2960434
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2448120, 9.2509537
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7889709, 13.7939110
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2930870, 19.3060760
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2110291, 13.2146149
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9315262, 14.9372711
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5534210, 13.5551033
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6403198, 14.6410904
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2707138, 13.2645531
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5942841, 16.5899200
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0790901, 14.0759583
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9486694, 12.9484138
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0580750, 15.0613518
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9220886, 18.9258728
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2374420, 18.2382431
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0244026, 16.0246010
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3597984, 14.3581715
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6691284, 16.6666908
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2895927, 17.2872963

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4938201, upper bound: 12.5369939
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5360605, upper bound: 12.4947640
time: 7.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8667450, 13.8685112
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4972343, 8.4940300
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4597015, 13.4601326
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0328522, 12.0296974
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6783447, 14.6749039
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1111717, 15.0995255
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6091003, 13.6041031
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0943298, 12.0949535
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9630775, 12.9700317
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9544754, 20.9583054
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0272713, 15.0270004
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5824585, 16.5858383
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7045670, 26.7100754
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7919846, 14.7938614
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2220383, 17.2226257
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4207268, 14.4187298
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5371170, 14.5359650
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6148376, 12.6126099
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9370575, 14.9366875
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2914543, 14.2951927
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2482643, 9.2474976
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7909241, 13.7919655
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2951927, 19.3039627
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2141380, 13.2115059
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9330673, 14.9357224
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5542984, 13.5542259
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6428146, 14.6385956
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2709122, 13.2643547
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5942993, 16.5899048
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0819664, 14.0730743
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9508362, 12.9462433
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0603638, 15.0590630
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9221115, 18.9258499
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2420654, 18.2336273
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0247688, 16.0242348
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585701, 14.3594036
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6692047, 16.6666298
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2874336, 17.2894592

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4837368, upper bound: 12.5470883
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5259670, upper bound: 12.5048481
time: 8.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8696098, 13.8656464
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4967804, 8.4944839
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4604645, 13.4593735
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0332336, 12.0293198
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6750183, 14.6782341
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1074295, 15.1032677
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6083755, 13.6048317
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0962219, 12.0930538
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9678116, 12.9652977
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9585342, 20.9542465
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0278969, 15.0263748
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5830383, 16.5852280
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7083511, 26.7063141
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7961349, 14.7896881
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2255096, 17.2191467
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4182167, 14.4212036
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5350952, 14.5379562
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6125717, 12.6148758
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9342804, 14.9394341
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2928352, 14.2938080
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2453308, 9.2504196
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7901840, 13.7927017
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2955055, 19.3036499
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2109947, 13.2146454
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9326706, 14.9361267
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5534706, 13.5550461
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6395493, 14.6418571
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2678719, 13.2673950
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5899277, 16.5942764
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0765038, 14.0785370
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9446564, 12.9524269
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0546036, 15.0648270
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9215622, 18.9263458
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2313385, 18.2443542
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0190315, 16.0299683
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3598137, 14.3581409
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6674042, 16.6684227
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2908211, 17.2859192

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5054960, upper bound: 12.5225265
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5477364, upper bound: 12.4802966
time: 9.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8669434, 13.8683090
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4962158, 8.4950485
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4601135, 13.4597168
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0327759, 12.0297699
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6764526, 14.6767960
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1074829, 15.1032143
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6078491, 13.6053543
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.0961990, 12.0930767
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9638062, 12.9693031
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9572449, 20.9555359
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0255890, 15.0286827
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.5802994, 16.5879669
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7062149, 26.7084274
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7922440, 14.7935791
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2232513, 17.2214050
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4205513, 14.4188671
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5369034, 14.5361519
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6140442, 12.6134033
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9367676, 14.9369469
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2936897, 14.2929573
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2487869, 9.2469635
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7921295, 13.7907562
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2976112, 19.3015442
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2141037, 13.2115364
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9342117, 14.9345779
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5543480, 13.5541725
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6420441, 14.6393661
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2680740, 13.2671928
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5899429, 16.5942612
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0793877, 14.0756569
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9468231, 12.9502563
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0568924, 15.0625381
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9215851, 18.9263229
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2359619, 18.2397385
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0193977, 16.0296059
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3585815, 14.3593731
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6674652, 16.6683617
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2886620, 17.2880821

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4954127, upper bound: 12.5326215
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5376428, upper bound: 12.4903810
time: 6.79 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4903810, upper bound: 12.5376428
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5326215, upper bound: 12.4954127
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4802967, upper bound: 12.5477363
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5225265, upper bound: 12.5054960
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5048481, upper bound: 12.5259669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5470884, upper bound: 12.4837368
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4947640, upper bound: 12.5360605
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5369939, upper bound: 12.4938201
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4799087, upper bound: 12.5544536
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5221496, upper bound: 12.5122261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4698247, upper bound: 12.5645319
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5120547, upper bound: 12.5223091
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4929306, upper bound: 12.5399922
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5351711, upper bound: 12.4977621
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4828465, upper bound: 12.5500857
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5250763, upper bound: 12.5078453
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5052777, upper bound: 12.5280657
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5475182, upper bound: 12.4858359
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4951938, upper bound: 12.5381594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5374239, upper bound: 12.4959194
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5197422, upper bound: 12.5150461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5619646, upper bound: 12.4728162
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5096588, upper bound: 12.5251399
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5518859, upper bound: 12.4828996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4912470, upper bound: 12.5400075
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5334878, upper bound: 12.4977778
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4811630, upper bound: 12.5501012
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5233931, upper bound: 12.5078612
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5029238, upper bound: 12.5255418
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5451646, upper bound: 12.4833121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4928397, upper bound: 12.5356360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5350699, upper bound: 12.4933958
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4933958, upper bound: 12.5350698
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5356361, upper bound: 12.4928397
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4833122, upper bound: 12.5451646
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5255418, upper bound: 12.5029237
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5078612, upper bound: 12.5233931
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5501012, upper bound: 12.4811630
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4977779, upper bound: 12.5334878
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5400076, upper bound: 12.4912469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4828996, upper bound: 12.5518859
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5251399, upper bound: 12.5096586
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4728162, upper bound: 12.5619645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5150461, upper bound: 12.5197422
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4959195, upper bound: 12.5374238
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5381594, upper bound: 12.4951938
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4858360, upper bound: 12.5475181
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5280657, upper bound: 12.5052776
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5078454, upper bound: 12.5250763
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5500858, upper bound: 12.4828465
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4977621, upper bound: 12.5351711
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5399922, upper bound: 12.4929305
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5223092, upper bound: 12.5120546
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5645319, upper bound: 12.4698246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5122261, upper bound: 12.5221495
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5544536, upper bound: 12.4799087
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4938201, upper bound: 12.5369939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5360605, upper bound: 12.4947640
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4837368, upper bound: 12.5470883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5259670, upper bound: 12.5048481
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5054960, upper bound: 12.5225265
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5477364, upper bound: 12.4802966
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.4954127, upper bound: 12.5326215
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.65
Output dim: 14, lower bound: -12.5376428, upper bound: 12.4903810

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8611450, 13.8599167
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4888535, 8.4889660
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4588585, 13.4595718
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0288849, 12.0335922
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6766129, 14.6746330
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1144791, 15.1196098
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6104584, 13.6128387
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1060715, 12.1078186
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9757462, 12.9714508
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9523544, 20.9555435
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0253410, 15.0233231
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6236115, 16.6186256
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7054749, 26.7073898
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7910538, 14.7886696
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2204857, 17.2238541
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4218445, 14.4221478
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5323715, 14.5320473
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6001358, 12.6008759
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9387894, 14.9393501
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2949677, 14.2965927
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2449226, 9.2459145
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7874184, 13.7884178
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2595253, 19.2524719
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2156982, 13.2172356
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9159660, 14.9178238
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5552864, 13.5556183
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6448364, 14.6464996
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2829132, 13.2853737
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5608521, 16.5500984
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0786476, 14.0746346
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9540253, 12.9467773
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0465469, 15.0402832
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9295235, 18.9211044
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2323723, 18.2243118
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0268593, 16.0121307
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3948784, 14.3883667
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6910477, 16.6865387
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3172607, 17.3170624

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4798057, upper bound: 12.5372840
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4900222, upper bound: 12.5270683
time: 6.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8602867, 13.8607750
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4875603, 8.4902554
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4586906, 13.4597359
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0300446, 12.0324287
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6763992, 14.6748466
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1156158, 15.1184731
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6098480, 13.6134529
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1036835, 12.1102142
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9764175, 12.9707756
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9525299, 20.9553680
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0265465, 15.0221214
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6282043, 16.6140366
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7085266, 26.7043533
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7885895, 14.7911301
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2207680, 17.2235756
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4206848, 14.4233055
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5318069, 14.5326157
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6009445, 12.6000671
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9408798, 14.9372520
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2947159, 14.2968445
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2446289, 9.2462120
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7870369, 13.7887955
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2569160, 19.2550812
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2152977, 13.2176399
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9185371, 14.9152565
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5554085, 13.5554962
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6441498, 14.6471863
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2850456, 13.2832413
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5569000, 16.5540428
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0728111, 14.0804672
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9528885, 12.9479141
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0485077, 15.0383301
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9275551, 18.9230728
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2313271, 18.2253571
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0256996, 16.0132904
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3890991, 14.3941479
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6881638, 16.6894341
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3150787, 17.3192406

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5220469, upper bound: 12.4950539
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5322627, upper bound: 12.4848375
time: 6.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8584824, 13.8625793
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4882889, 8.4895267
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4585075, 13.4599152
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0284271, 12.0340424
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6780472, 14.6731987
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1145325, 15.1195564
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6099396, 13.6133614
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1060486, 12.1078453
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9717369, 12.9754562
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9510651, 20.9568405
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0230331, 15.0256310
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6208725, 16.6213684
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7033691, 26.7095108
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7871552, 14.7925644
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2182274, 17.2261086
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4241791, 14.4198112
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5341873, 14.5302429
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6016045, 12.5994072
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9412766, 14.9368629
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2958221, 14.2957382
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2483788, 9.2424622
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7893639, 13.7864685
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2616310, 19.2503662
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2188072, 13.2141266
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9175148, 14.9162788
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5561600, 13.5547447
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6473312, 14.6440086
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2831116, 13.2851715
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5608673, 16.5500832
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0815239, 14.0717506
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9561920, 12.9446068
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0488358, 15.0379944
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9295464, 18.9210815
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2369881, 18.2196960
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0272255, 16.0117645
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3936501, 14.3895988
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6911087, 16.6864777
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3150940, 17.3192253

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4697215, upper bound: 12.5473775
time: 12.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4799379, upper bound: 12.5371619
time: 6.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8576241, 13.8634377
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4869995, 8.4908161
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4583397, 13.4600830
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0296021, 12.0328789
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6778336, 14.6734123
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1156693, 15.1184196
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6093292, 13.6139755
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1036530, 12.1102371
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9724121, 12.9747810
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9512405, 20.9566650
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0242348, 15.0244293
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6254578, 16.6167793
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7064209, 26.7064667
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7846985, 14.7950249
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2185097, 17.2258339
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4230194, 14.4209690
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5336227, 14.5308113
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6024170, 12.5985947
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9433670, 14.9347649
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2955704, 14.2959900
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2480850, 9.2427559
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7889900, 13.7868462
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2590294, 19.2529678
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2184067, 13.2145309
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9200859, 14.9137115
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5562859, 13.5546188
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6466446, 14.6446915
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2852478, 13.2830391
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5569153, 16.5540276
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0756950, 14.0775833
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9550552, 12.9457397
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0507965, 15.0360413
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9275780, 18.9230499
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2359505, 18.2207336
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0260658, 16.0129242
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3878670, 14.3953781
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6882248, 16.6893730
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3129196, 17.3213997

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5119520, upper bound: 12.5051372
time: 17.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5221677, upper bound: 12.4949208
time: 12.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8613472, 13.8597145
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4878349, 8.4899807
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4592705, 13.4591560
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0288086, 12.0336685
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6747208, 14.6765251
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1107903, 15.1232986
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6092072, 13.6140938
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1079407, 12.1059456
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9764709, 12.9707222
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9551239, 20.9527740
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0236588, 15.0250053
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6214828, 16.6207886
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7071228, 26.7057419
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7913361, 14.7884102
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2217064, 17.2226334
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4217072, 14.4223232
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5321884, 14.5322647
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5993423, 12.6016693
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9385147, 14.9396400
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2972031, 14.2943573
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2454605, 9.2453957
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7886238, 13.7872086
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2619438, 19.2500610
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2156677, 13.2172699
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9171104, 14.9166794
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5553398, 13.5555687
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6440735, 14.6472664
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2800751, 13.2882118
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5564880, 16.5544548
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0760612, 14.0772171
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9500122, 12.9507904
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0430756, 15.0437584
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9290504, 18.9216309
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2262611, 18.2304230
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0214882, 16.0175018
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3949089, 14.3883553
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6893234, 16.6882706
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3186340, 17.3158379

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4942729, upper bound: 12.5256081
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5044893, upper bound: 12.5153924
time: 13.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8604851, 13.8605728
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4865456, 8.4912739
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591026, 13.4593239
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0299683, 12.0325050
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6745071, 14.6767426
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1119270, 15.1221619
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6085968, 13.6147041
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1055527, 12.1083374
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9771461, 12.9700470
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9552994, 20.9525986
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0248642, 15.0237999
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6260757, 16.6161995
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7101746, 26.7027054
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7888794, 14.7908707
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2219887, 17.2223549
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4205475, 14.4234810
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5316238, 14.5328331
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6001511, 12.6008606
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9406204, 14.9375420
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2969513, 14.2946091
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2451668, 9.2456894
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7882500, 13.7875862
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2593346, 19.2526627
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2152634, 13.2176704
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9196815, 14.9141121
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5554619, 13.5554466
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6433868, 14.6479530
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2822075, 13.2860794
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5525513, 16.5583992
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0702248, 14.0830498
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9488754, 12.9519272
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0450287, 15.0418015
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9270821, 18.9235992
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2252235, 18.2314606
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0203285, 16.0186615
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3891296, 14.3941345
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6864243, 16.6911659
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3164597, 17.3180161

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5365124, upper bound: 12.4833780
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5467296, upper bound: 12.4731617
time: 16.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8586845, 13.8623772
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4872704, 8.4905453
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4589272, 13.4595032
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0283585, 12.0341187
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6761551, 14.6750908
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1108437, 15.1232452
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6086884, 13.6146164
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1079254, 12.1059723
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9724655, 12.9747276
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9538269, 20.9540710
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0213509, 15.0273132
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6187439, 16.6235275
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7050171, 26.7078629
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7874451, 14.7923050
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2194481, 17.2248878
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4240417, 14.4199867
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5339890, 14.5304604
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6008110, 12.6002007
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9410019, 14.9371529
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2980576, 14.2935028
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2489128, 9.2419395
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7905693, 13.7852631
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2640495, 19.2479477
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2187767, 13.2141609
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9186592, 14.9151344
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5562172, 13.5546913
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6465607, 14.6447754
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2802734, 13.2880135
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5565033, 16.5544395
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0789452, 14.0743332
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9521790, 12.9486198
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0453644, 15.0414696
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9290733, 18.9216080
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2308769, 18.2258072
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0218544, 16.0171356
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3936806, 14.3895855
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6893845, 16.6882095
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3164749, 17.3180008

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4841889, upper bound: 12.5357017
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4944052, upper bound: 12.5254860
time: 8.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8578224, 13.8632355
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4859810, 8.4918346
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4587593, 13.4596710
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0295258, 12.0329552
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6759415, 14.6753044
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1119804, 15.1221085
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6080780, 13.6152267
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1055298, 12.1083641
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9731407, 12.9740524
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9540024, 20.9538956
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0225563, 15.0261116
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6233368, 16.6189423
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7080688, 26.7048187
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7849808, 14.7947655
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2197304, 17.2246132
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4228821, 14.4211464
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5334244, 14.5310249
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6016197, 12.5993881
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9431076, 14.9350548
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2978058, 14.2937546
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2486191, 9.2422371
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7901955, 13.7856369
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2614479, 19.2505493
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2183723, 13.2145653
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9212303, 14.9125671
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5563393, 13.5545692
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6458817, 14.6454582
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2824059, 13.2858810
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5525665, 16.5583839
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0731087, 14.0801659
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9510422, 12.9497528
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0473175, 15.0395126
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9271049, 18.9235764
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2298393, 18.2268448
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0206947, 16.0182953
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3878975, 14.3953667
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6864853, 16.6911049
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3143005, 17.3201752

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5264193, upper bound: 12.4934613
time: 17.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5366351, upper bound: 12.4832450
time: 11.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8608818, 13.8601799
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4914703, 8.4863453
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4586525, 13.4598007
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0318909, 12.0306206
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6780930, 14.6731606
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1242447, 15.1098442
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6144562, 13.6088486
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1060486, 12.1078491
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9722786, 12.9749146
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9499512, 20.9579544
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0266228, 15.0220451
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6214447, 16.6208916
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7017517, 26.7111359
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7917175, 14.7880516
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2203407, 17.2239990
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4222412, 14.4217854
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5326157, 14.5317879
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6015625, 12.5994492
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9381180, 14.9403000
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2917099, 14.2998505
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2433281, 9.2475662
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7856407, 13.7901955
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2515297, 19.2604828
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2164726, 13.2164650
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9131126, 14.9206810
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5544853, 13.5564308
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6468277, 14.6445084
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2882309, 13.2800560
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5646667, 16.5462799
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0849953, 14.0682831
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9565277, 12.9442711
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0471039, 15.0397301
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9273720, 18.9234962
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2371483, 18.2195282
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0271416, 16.0118484
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3953285, 14.3879471
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6937180, 16.6838799
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3179092, 17.3168259

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4693335, upper bound: 12.5540945
time: 18.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4795499, upper bound: 12.5438656
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8600197, 13.8610382
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4901810, 8.4876347
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4584846, 13.4599686
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0330582, 12.0294571
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6778793, 14.6733780
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1253815, 15.1087074
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6138458, 13.6094589
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1036530, 12.1102409
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9729500, 12.9742432
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9501190, 20.9577789
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0278282, 15.0208435
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6260376, 16.6163025
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7047882, 26.7080917
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7892609, 14.7905121
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2206230, 17.2237206
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4210892, 14.4229431
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5320511, 14.5323563
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6023712, 12.5986404
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9402237, 14.9382019
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2914581, 14.3001022
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2430344, 9.2478600
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7852669, 13.7905731
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2489281, 19.2630920
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2160683, 13.2168694
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9156761, 14.9181137
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5546074, 13.5563049
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6461411, 14.6451950
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2903633, 13.2779236
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5607147, 16.5502243
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0791588, 14.0741158
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9553909, 12.9454079
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0490646, 15.0377731
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9254036, 18.9254684
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2361107, 18.2205734
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0259819, 16.0130081
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3895454, 14.3937283
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6908188, 16.6867752
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3157349, 17.3190002

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5115749, upper bound: 12.5118672
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5217908, upper bound: 12.5016511
time: 7.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8582153, 13.8628426
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4909096, 8.4869099
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4583092, 13.4601479
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0314484, 12.0310707
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6795273, 14.6717262
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1242981, 15.1097908
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6139297, 13.6093712
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1060181, 12.1078720
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9682732, 12.9789200
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9486542, 20.9592438
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0243149, 15.0243568
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6187057, 16.6236343
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.6996307, 26.7132492
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7878265, 14.7919464
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2180824, 17.2262535
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4245834, 14.4194489
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5344162, 14.5299835
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6030312, 12.5979805
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9406052, 14.9378128
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2925644, 14.2989960
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2467842, 9.2441139
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7875862, 13.7882500
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2536430, 19.2583771
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2195816, 13.2133598
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9146614, 14.9191360
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5553589, 13.5555534
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6493225, 14.6420174
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2884293, 13.2798576
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5646820, 16.5462646
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0878792, 14.0654030
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9587021, 12.9421005
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0493927, 15.0374413
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9274025, 18.9234734
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2417717, 18.2149200
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0275078, 16.0114822
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3940964, 14.3891792
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6937790, 16.6838188
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3157501, 17.3189888

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4592495, upper bound: 12.5641726
time: 14.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4694658, upper bound: 12.5539354
time: 23.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8573570, 13.8637047
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4896202, 8.4881992
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4581413, 13.4603119
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0326080, 12.0299072
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6793137, 14.6719398
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1254349, 15.1086540
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6133194, 13.6099815
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1036224, 12.1102676
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9689445, 12.9782486
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9488297, 20.9590683
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0255165, 15.0231514
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6232910, 16.6190414
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7026672, 26.7102051
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7853622, 14.7944069
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2183647, 17.2259789
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4234238, 14.4206066
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5338516, 14.5305519
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6038437, 12.5971680
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9427109, 14.9357147
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2923050, 14.2992477
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2464905, 9.2444038
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7872124, 13.7886238
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2510338, 19.2609787
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2191772, 13.2137604
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9172249, 14.9165649
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5554810, 13.5554314
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6486359, 14.6427002
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2905617, 13.2777252
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5607452, 16.5502090
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0820427, 14.0712357
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9575653, 12.9432373
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0513535, 15.0354843
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9254265, 18.9254456
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2407265, 18.2159576
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0263481, 16.0126419
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3883171, 14.3949604
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6908798, 16.6867142
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3135757, 17.3211632

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5014801, upper bound: 12.5219503
time: 19.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5116959, upper bound: 12.5117340
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8610802, 13.8599777
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4904556, 8.4873638
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4590645, 13.4593887
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0318146, 12.0306969
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6762009, 14.6750526
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1205559, 15.1135330
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6132050, 13.6101036
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1079178, 12.1059723
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9730034, 12.9741898
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9527130, 20.9551849
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0249443, 15.0237274
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6192856, 16.6230431
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7033997, 26.7094879
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7919769, 14.7877731
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2215614, 17.2227821
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4220657, 14.4219227
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5324020, 14.5320015
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6007652, 12.6002464
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9378281, 14.9405785
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2939453, 14.2976151
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2438660, 9.2470284
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7868462, 13.7889900
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2539482, 19.2580719
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2164383, 13.2164955
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9142570, 14.9195366
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5545349, 13.5563736
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6460648, 14.6452751
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2853889, 13.2828979
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5603180, 16.5506363
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0824089, 14.0708656
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9525146, 12.9482841
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0436325, 15.0432053
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9268456, 18.9239578
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2310448, 18.2256470
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0217705, 16.0172157
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3953400, 14.3879166
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6919785, 16.6856117
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3191376, 17.3154526

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4823553, upper bound: 12.5396334
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4925718, upper bound: 12.5294177
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8602219, 13.8608398
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4891624, 8.4886532
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4588966, 13.4595528
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0329819, 12.0295334
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6759720, 14.6752701
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1216927, 15.1123962
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6125946, 13.6107140
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1055298, 12.1083679
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9736786, 12.9735146
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9528885, 20.9550095
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0261459, 15.0225258
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6238708, 16.6184540
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7064362, 26.7064438
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7895203, 14.7902336
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2218437, 17.2224998
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4209137, 14.4230804
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5318375, 14.5325699
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6015778, 12.5994339
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9399338, 14.9384766
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2936935, 14.2978668
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2435722, 9.2473259
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7864723, 13.7893639
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2513466, 19.2606735
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2160339, 13.2168999
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9168205, 14.9169693
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5546570, 13.5562515
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6453781, 14.6459618
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2875214, 13.2807655
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5563660, 16.5545807
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0765800, 14.0766983
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9513779, 12.9494209
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0455856, 15.0412483
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9248772, 18.9259262
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2299995, 18.2266846
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0206108, 16.0183792
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3895607, 14.3936977
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6890793, 16.6885071
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3169632, 17.3176270

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5245966, upper bound: 12.4974033
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5348123, upper bound: 12.4871871
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8584175, 13.8626442
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4898911, 8.4879246
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4587212, 13.4597321
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0313721, 12.0311470
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6776352, 14.6736183
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1206093, 15.1134796
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6126785, 13.6106224
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1078949, 12.1059990
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9689980, 12.9781952
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9514236, 20.9564819
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0226326, 15.0260353
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6165466, 16.6257858
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7012787, 26.7116013
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7880859, 14.7916679
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2193031, 17.2250328
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4244080, 14.4195862
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5342026, 14.5301971
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6022377, 12.5987740
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9403152, 14.9380913
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2947998, 14.2967606
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2473183, 9.2435760
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7887993, 13.7870407
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2560616, 19.2559586
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2195473, 13.2133904
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9158058, 14.9179916
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5554085, 13.5555000
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6485519, 14.6427841
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2855911, 13.2826958
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5603333, 16.5506210
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0852928, 14.0679817
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9546890, 12.9461136
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0459213, 15.0409164
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9268684, 18.9239349
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2356606, 18.2210236
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0221367, 16.0168495
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3941078, 14.3891487
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6920395, 16.6855507
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3169785, 17.3176117

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4722714, upper bound: 12.5497269
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4824877, upper bound: 12.5395113
time: 6.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8575592, 13.8635025
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4886017, 8.4892139
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4585533, 13.4598999
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0325317, 12.0299835
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6774216, 14.6738358
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1217461, 15.1123428
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6120682, 13.6112366
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1054993, 12.1083908
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9696732, 12.9775200
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9515991, 20.9563065
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0238380, 15.0248337
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6211319, 16.6211967
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7043152, 26.7085571
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7856216, 14.7941246
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2195854, 17.2247581
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4232483, 14.4207439
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5336380, 14.5307655
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6030464, 12.5979652
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9424210, 14.9359894
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2945480, 14.2970123
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2470245, 9.2438698
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7884178, 13.7874184
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2534523, 19.2585602
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2191429, 13.2137947
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9183693, 14.9154205
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5555344, 13.5553741
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6478729, 14.6434669
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2877197, 13.2805634
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5563812, 16.5545654
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0794563, 14.0738182
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9535522, 12.9472504
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0478745, 15.0389595
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9249001, 18.9259033
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2346153, 18.2220612
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0209770, 16.0180130
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3883286, 14.3949299
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6891403, 16.6884460
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3148041, 17.3197861

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5145017, upper bound: 12.5074865
time: 11.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5247175, upper bound: 12.4972702
time: 14.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8619385, 13.8591576
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4888153, 8.4890003
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4591637, 13.4592857
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0290527, 12.0334854
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6754532, 14.6758461
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1131477, 15.1208687
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6104507, 13.6128502
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1067810, 12.1071510
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9766197, 12.9705734
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9541702, 20.9538040
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0251999, 15.0234718
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6232758, 16.6190491
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7068024, 26.7060852
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7918930, 14.7878571
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2223778, 17.2220421
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4211273, 14.4228630
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5316391, 14.5327606
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5991554, 12.6018791
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9381180, 14.9403229
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2957687, 14.2957916
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2447357, 9.2461662
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7874260, 13.7884140
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2595406, 19.2524719
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2147827, 13.2181511
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9159813, 14.9178123
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5553246, 13.5555840
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6440506, 14.6472893
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2817841, 13.2865028
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5585938, 16.5523453
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0769691, 14.0763092
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9515076, 12.9493408
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0437241, 15.0432281
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9287453, 18.9220619
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2280159, 18.2288513
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0236092, 16.0153809
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3949318, 14.3883247
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6899185, 16.6876793
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3182449, 17.3163452

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4947025, upper bound: 12.5277068
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5049189, upper bound: 12.5174912
time: 10.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8610764, 13.8600159
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4875221, 8.4902935
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4589958, 13.4594536
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0302124, 12.0323181
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6752396, 14.6760597
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1142845, 15.1197357
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6098404, 13.6134644
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1043854, 12.1095428
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9772949, 12.9698982
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9543457, 20.9536285
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0264015, 15.0222664
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6278687, 16.6144600
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7098389, 26.7030411
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7894287, 14.7903175
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2226524, 17.2217674
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4199677, 14.4240208
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5310745, 14.5333290
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.5999641, 12.6010704
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9402237, 14.9382248
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2955093, 14.2960472
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2444420, 9.2464600
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7870445, 13.7887878
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2569389, 19.2550812
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2143822, 13.2185555
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9185524, 14.9152451
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5554466, 13.5554619
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6433640, 14.6479759
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2839165, 13.2843704
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5546570, 16.5562935
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0711327, 14.0821419
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9503708, 12.9504776
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0456848, 15.0412750
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9267769, 18.9240341
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2269783, 18.2298965
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0224495, 16.0165405
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3891525, 14.3941040
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6870193, 16.6905708
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3160706, 17.3185196

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1295

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5369436, upper bound: 12.4854771
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5471594, upper bound: 12.4752609
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.8592720, 13.8618202
1: -3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.4882507, 8.4895649
2: -0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4588203, 13.4596329
3: -1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0285950, 12.0339355
4: -11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6768875, 14.6744118
5: 1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796
6: -39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.1132011, 15.1208191
7: -3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6099243, 13.6133766
8: -6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1067505, 12.1071777
9: -4.8001633, 11.7221231, -4.8001633, 11.7221231, -12.9726143, 12.9745789
10: 1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9528732, 20.9551010
11: -11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476
12: -11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0228920, 15.0257797
13: -18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6205368, 16.6217880
14: 4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7046814, 26.7081985
15: -8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198
16: -16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.7879944, 14.7917519
17: 6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2201195, 17.2243004
18: -14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4234619, 14.4205284
19: -20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5334396, 14.5309563
20: -2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6006241, 12.6004066
21: -11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549
22: -3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9406052, 14.9378357
23: -14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.2966156, 14.2949409
24: -19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2481918, 9.2427101
25: -5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.7893715, 13.7864647
26: -21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.2616539, 19.2503662
27: -16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2178917, 13.2150459
28: -12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141
29: -5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9175301, 14.9162636
30: -10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5561981, 13.5547066
31: -10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6465378, 14.6447945
32: -24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.2819824, 13.2863007
33: -69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.5586090, 16.5523300
34: -53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.0798531, 14.0734291
35: -47.8248253, -26.0548592, -47.8248253, -26.0548592, -12.9536819, 12.9471664
36: -42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.0460129, 15.0409393
37: -86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9287682, 18.9220390
38: -52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.2326317, 18.2242355
39: -76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0239754, 16.0150146
40: -67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3937035, 14.3895550
41: -55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.6899796, 16.6876183
42: -29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.3160858, 17.3185043

Time for backsubstitution: 2.22 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 24.22 + 1776.41 = 1800.63 seconds
