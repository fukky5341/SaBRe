## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 7)
Time budget: 7200 seconds
Split limit: 100
Threshold: 45.0013539996


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3955650, 44.3955612)
1: (-13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096)
2: (-14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4909210, 35.4909248)
3: (-12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700)
4: (-21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189)
5: (-12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787)
6: (-50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5662613, 40.5662537)
7: (-16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798)
8: (-18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384)
9: (-16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6461983, 38.6462021)
10: (-24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8275909, 61.8275909)
11: (-24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975)
12: (-28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9375648, 46.9375610)
13: (-32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388)
14: (-23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1073761, 60.1073761)
15: (-18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818)
16: (-32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851)
17: (-17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2744904, 55.2744942)
18: (-25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630)
19: (-26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995)
20: (-21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145)
21: (-25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867)
22: (-22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133)
23: (-21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798)
24: (-32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783)
25: (-18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148)
26: (-29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081)
27: (-32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254776, 47.8254814)
28: (-21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660)
29: (-23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474)
30: (-29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9555359, 45.9555321)
31: (-26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585)
32: (-42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5713501, 47.5713501)
33: (-72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3344040, 61.3344116)
34: (-56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6207123, 43.6207085)
35: (-50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2550659, 48.2550697)
36: (-47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0323868, 52.0323792)
37: (-83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4293747, 58.4293747)
38: (-58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3017960, 61.3017883)
39: (-78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3207550, 65.3207550)
40: (-67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1823769, 41.1823730)
41: (-55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2860641, 42.2860641)
42: (-33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7043839, 37.7043839)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.93 + 60.50 = 63.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -45.0464004, upper bound: 45.0464004

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0030915, upper bound: 45.0448113
time: 54.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0448113, upper bound: 45.0030915
time: 32.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 86.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 86.75
Output dim: 14, lower bound: -45.0030915, upper bound: 45.0448113
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 86.75
Output dim: 14, lower bound: -45.0448113, upper bound: 45.0030915

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3845596, 44.3867874
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4845848, 35.4860001
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5815811, 40.5892181
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6507797, 38.6497154
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8446960, 61.8407249
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9402885, 46.9395218
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1164818, 60.1145287
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2908325, 55.2872658
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8428497, 47.8479500
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9499741, 45.9482307
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5685844, 47.5688896
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3265686, 61.3259811
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6048126, 43.6021805
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2323227, 48.2290535
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0368042, 52.0379181
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4302826, 58.4304390
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3192902, 61.3240356
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3149490, 65.3149338
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1716385, 41.1764984
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2998810, 42.3041496
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7304802, 37.7362900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9669113, upper bound: 45.0426349
time: 51.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0015599, upper bound: 45.0086171
time: 27.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3867722, 44.3845673
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4859962, 35.4845810
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5892181, 40.5815811
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6497192, 38.6507835
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8407288, 61.8446884
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9395180, 46.9402924
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1145287, 60.1164780
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2872620, 55.2908401
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8479462, 47.8428497
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9482269, 45.9499741
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5688896, 47.5685806
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3259888, 61.3265610
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6021805, 43.6048126
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2290573, 48.2323227
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0379181, 52.0367966
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4304352, 58.4302940
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3240356, 61.3192825
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3149338, 65.3149567
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1764984, 41.1716385
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.3041458, 42.2998848
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7362938, 37.7304764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0086171, upper bound: 45.0015599
time: 29.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0426349, upper bound: 44.9669113
time: 48.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 80.17 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 80.17
Output dim: 14, lower bound: -44.9669113, upper bound: 45.0426349
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 80.17
Output dim: 14, lower bound: -45.0015599, upper bound: 45.0086171
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 80.17
Output dim: 14, lower bound: -45.0086171, upper bound: 45.0015599
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 80.17
Output dim: 14, lower bound: -45.0426349, upper bound: 44.9669113

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3821182, 44.3854752
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4821587, 35.4847374
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5813828, 40.5890656
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6507874, 38.6497154
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8461304, 61.8400764
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9381447, 46.9356918
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1153145, 60.1135941
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2917557, 55.2870941
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8422394, 47.8502007
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9488716, 45.9462357
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5669022, 47.5667114
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3223572, 61.3179779
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5980835, 43.5896797
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2278290, 48.2203979
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0365753, 52.0377121
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4265518, 58.4261398
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3190460, 61.3246078
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3122864, 65.3114700
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1704788, 41.1750450
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2983894, 42.3024101
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7303276, 37.7363205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9297467, upper bound: 45.0414460
time: 34.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9657191, upper bound: 45.0055415
time: 46.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3832474, 44.3843422
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4833183, 35.4835739
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5813828, 40.5890198
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6507797, 38.6497192
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8440552, 61.8421631
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9364586, 46.9374008
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1155281, 60.1133614
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2906647, 55.2881813
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8451080, 47.8473434
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9479790, 45.9471283
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5663986, 47.5671349
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3185577, 61.3217697
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5923080, 43.5954514
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2236633, 48.2241859
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0365906, 52.0376816
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4259872, 58.4266891
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3198700, 61.3237839
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3114929, 65.3122711
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1701889, 41.1753387
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2981453, 42.3026466
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7305107, 37.7361450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9644241, upper bound: 45.0074268
time: 31.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -45.0003689, upper bound: 44.9715467
time: 26.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3843307, 44.3832550
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4835777, 35.4833221
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5890198, 40.5813904
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6497192, 38.6507835
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8421631, 61.8440399
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9373970, 46.9364624
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1133614, 60.1155205
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2881775, 55.2906685
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8473358, 47.8451004
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9471245, 45.9479790
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5671310, 47.5664024
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3217773, 61.3185577
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5954514, 43.5923080
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2241821, 48.2236671
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0376740, 52.0365906
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4266891, 58.4259949
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3237762, 61.3198776
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3122711, 65.3114929
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1753311, 41.1701851
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.3026466, 42.2981453
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7361412, 37.7305069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9715467, upper bound: 45.0003689
time: 52.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0074268, upper bound: 44.9644241
time: 65.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3854752, 44.3821220
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4847374, 35.4821587
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5890579, 40.5813904
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6497192, 38.6507835
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8400879, 61.8461266
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9356956, 46.9381485
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1135902, 60.1153107
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2870865, 55.2917557
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8502045, 47.8422432
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9462395, 45.9488716
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5667114, 47.5668983
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3179779, 61.3223495
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5896759, 43.5980835
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2203979, 48.2278290
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0377045, 52.0365677
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4261398, 58.4265556
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3246002, 61.3190308
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3114624, 65.3122864
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1750412, 41.1704788
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.3024178, 42.2983818
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7363243, 37.7303314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0055415, upper bound: 44.9657191
time: 26.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0414460, upper bound: 44.9297467
time: 56.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 85.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 85.36
Output dim: 14, lower bound: -44.9297467, upper bound: 45.0414460
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 85.36
Output dim: 14, lower bound: -44.9657191, upper bound: 45.0055415
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 85.36
Output dim: 14, lower bound: -44.9644241, upper bound: 45.0074268
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 85.36
Output dim: 14, lower bound: -45.0003689, upper bound: 44.9715467
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 85.36
Output dim: 14, lower bound: -44.9715467, upper bound: 45.0003689
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 85.36
Output dim: 14, lower bound: -45.0074268, upper bound: 44.9644241
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 85.36
Output dim: 14, lower bound: -45.0055415, upper bound: 44.9657191
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 85.36
Output dim: 14, lower bound: -45.0414460, upper bound: 44.9297467

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3757935, 44.3799477
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4773407, 35.4804840
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5584259, 40.5602150
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6300735, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6368790, 38.6386566
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8347397, 61.8312378
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9405556, 46.9373741
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0981293, 60.0999947
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2936401, 55.2888756
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8423920, 47.8503342
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9328728, 45.9261665
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5685501, 47.5682831
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3090363, 61.3003998
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5832596, 43.5708122
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2155838, 48.2044983
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0308838, 52.0302811
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4070892, 58.4003525
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3071671, 61.3092651
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3004684, 65.2955475
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1759338, 41.1767426
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2856293, 42.2857590
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7411499, 37.7502937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8736830, upper bound: 45.0410358
time: 37.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8736830, upper bound: 44.9674823
time: 47.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3765869, 44.3791542
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4779053, 35.4799156
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5525360, 40.5661049
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6326981, 39.6326981
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6397324, 38.6358109
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8373032, 61.8286819
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9398308, 46.9381027
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1017151, 60.0964127
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2935333, 55.2889862
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8423767, 47.8503418
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9287987, 45.9302330
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5684662, 47.5683632
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3047791, 61.3046722
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5792160, 43.5748596
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2119293, 48.2081528
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0291443, 52.0320129
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4007721, 58.4066849
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3036880, 61.3127518
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2963638, 65.2996521
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1721802, 41.1804962
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2817383, 42.2896500
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7443085, 37.7471390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8916037, upper bound: 45.0051142
time: 51.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9653070, upper bound: 44.9495886
time: 46.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3769302, 44.3788147
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4785004, 35.4793205
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5584259, 40.5601692
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6310577, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6368790, 38.6386604
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8326340, 61.8333282
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9388695, 46.9390793
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0983429, 60.0997658
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2925568, 55.2899628
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8452454, 47.8474770
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9319725, 45.9270592
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5680542, 47.5687065
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3052521, 61.3041916
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5774918, 43.5765839
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2114258, 48.2082863
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0308838, 52.0302505
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4065399, 58.4009018
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3080215, 61.3084412
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2996750, 65.2963486
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1756439, 41.1770325
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2853928, 42.2859955
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7413254, 37.7501144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9083672, upper bound: 45.0070157
time: 83.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9639970, upper bound: 44.9334586
time: 30.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3788147, 44.3769341
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4793243, 35.4785004
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5601730, 40.5584259
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6310577
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6386642, 38.6368752
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8333359, 61.8326454
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9390831, 46.9388695
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0997620, 60.0983391
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2899628, 55.2925606
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8474731, 47.8452454
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9270592, 45.9319763
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5687103, 47.5680542
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3041840, 61.3052521
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5765839, 43.5774918
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2082901, 48.2114258
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0302429, 52.0308914
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4008942, 58.4065399
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3084488, 61.3080292
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2963486, 65.2996750
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1770325, 41.1756363
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2860031, 42.2853851
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7501221, 37.7413254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9334586, upper bound: 44.9639970
time: 51.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0070157, upper bound: 44.9083672
time: 33.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3791504, 44.3765945
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4799194, 35.4779053
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5661011, 40.5525398
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6326981, 39.6326981
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6358109, 38.6397247
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8286819, 61.8372917
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9380989, 46.9398270
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0964050, 60.1017151
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2889862, 55.2935371
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8503418, 47.8423767
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9302330, 45.9288025
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5683594, 47.5684738
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3046722, 61.3047791
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5748596, 43.5792160
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2081528, 48.2119293
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0320129, 52.0291367
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4066925, 58.4007683
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3127518, 61.3036957
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2996597, 65.2963715
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1804962, 41.1721725
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2896576, 42.2817307
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7471313, 37.7443008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9495886, upper bound: 44.9653070
time: 56.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0051142, upper bound: 44.8916037
time: 69.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3799438, 44.3758011
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4804840, 35.4773369
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5602112, 40.5584297
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6300735
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6386642, 38.6368790
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8312302, 61.8347321
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9373741, 46.9405556
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0999908, 60.0981293
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2888641, 55.2936478
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8503418, 47.8423843
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9261665, 45.9328690
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5682831, 47.5685539
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3003998, 61.3090363
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5708084, 43.5832596
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2044983, 48.2155838
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0302734, 52.0308762
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4003448, 58.4071007
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3092728, 61.3071747
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2955551, 65.3004761
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1767426, 41.1759262
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2857666, 42.2856216
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7502899, 37.7411461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9674823, upper bound: 44.9293192
time: 22.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0410358, upper bound: 44.8736830
time: 78.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 103.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.8736830, upper bound: 45.0410358
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.8736830, upper bound: 44.9674823
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.8916037, upper bound: 45.0051142
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.9653070, upper bound: 44.9495886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.9083672, upper bound: 45.0070157
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.9639970, upper bound: 44.9334586
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.9334586, upper bound: 44.9639970
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.35
Output dim: 14, lower bound: -45.0070157, upper bound: 44.9083672
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.9495886, upper bound: 44.9653070
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.35
Output dim: 14, lower bound: -45.0051142, upper bound: 44.8916037
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 103.35
Output dim: 14, lower bound: -44.9674823, upper bound: 44.9293192
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.35
Output dim: 14, lower bound: -45.0410358, upper bound: 44.8736830

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3768539, 44.3809624
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4774094, 35.4805527
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5465622, 40.5450706
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6233063, 39.6302528
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6268196, 38.6311226
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8270416, 61.8253822
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9404793, 46.9372978
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0834427, 60.0889816
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2901382, 55.2859039
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8409576, 47.8489189
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9267235, 45.9179802
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5707283, 47.5706558
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2983856, 61.2862015
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5738144, 43.5582237
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2062836, 48.1921043
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0261765, 52.0240250
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3844681, 58.3702011
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2963562, 61.2947617
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2878799, 65.2787781
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1655235, 41.1620941
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2715149, 42.2668610
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7502747, 37.7616730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8425831, upper bound: 45.0399372
time: 32.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8720962, upper bound: 44.9948942
time: 20.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3776169, 44.3801689
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4779663, 35.4799881
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5403366, 40.5509605
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6259384, 39.6276131
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6296730, 38.6280937
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8295898, 61.8226509
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9397545, 46.9380226
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0870285, 60.0852356
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2900314, 55.2859726
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8409576, 47.8489265
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9224815, 45.9220428
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5706520, 47.5707359
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2941132, 61.2904739
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5697708, 43.5622673
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2026291, 48.1957626
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0244370, 52.0257568
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3781509, 58.3765335
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2927551, 61.2982407
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2837753, 65.2828827
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1615715, 41.1658478
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2675552, 42.2707520
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7534332, 37.7585144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8605095, upper bound: 45.0040322
time: 41.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8900212, upper bound: 44.9589991
time: 25.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3779984, 44.3798294
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4785690, 35.4793930
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5465622, 40.5450249
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6242905, 39.6291199
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6268196, 38.6311264
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8249512, 61.8274689
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9387932, 46.9389992
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0836563, 60.0887489
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2890549, 55.2869911
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8438110, 47.8460617
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9258385, 45.9188728
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5702248, 47.5710793
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2945862, 61.2899933
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5680466, 43.5639915
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2021255, 48.1958923
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0262070, 52.0240021
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3839188, 58.3707504
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2972107, 61.2939377
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2870865, 65.2795715
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1652260, 41.1623878
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2712784, 42.2670975
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7504578, 37.7614975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8617068, upper bound: 45.0053782
time: 26.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9075415, upper bound: 44.9776748
time: 41.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3798294, 44.3780022
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4793930, 35.4785690
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5450211, 40.5465622
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6291199, 39.6242943
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6311226, 38.6268196
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8274689, 61.8249550
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9389992, 46.9387932
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0887527, 60.0836525
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2869873, 55.2890549
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8460541, 47.8438225
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9188728, 45.9258347
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5710793, 47.5702324
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2899933, 61.2945938
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5639954, 43.5680428
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1958923, 48.2021255
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0240097, 52.0261993
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3707504, 58.3839188
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2939453, 61.2972031
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2795792, 65.2870789
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1623802, 41.1652260
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2670975, 42.2712784
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7614899, 37.7504578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9776748, upper bound: 44.9075415
time: 32.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0053782, upper bound: 44.8617068
time: 48.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3801651, 44.3776169
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4799881, 35.4779663
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5509567, 40.5403328
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6276169, 39.6259346
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6280937, 38.6296692
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8226624, 61.8296013
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9380226, 46.9397545
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0852432, 60.0870247
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2859726, 55.2900314
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8489227, 47.8409538
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9220390, 45.9224854
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5707283, 47.5706482
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2904663, 61.2941132
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5622711, 43.5697708
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1957626, 48.2026291
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0257492, 52.0244293
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3765335, 58.3781433
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2982483, 61.2927628
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2828751, 65.2837830
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1658440, 41.1615753
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2707520, 42.2675552
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7585144, 37.7534332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9589991, upper bound: 44.8900212
time: 128.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0040322, upper bound: 44.8605095
time: 47.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3809586, 44.3768654
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4805527, 35.4774055
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5450668, 40.5465660
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6302567, 39.6233063
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6311226, 38.6268196
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8253784, 61.8270416
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9372978, 46.9404793
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0889816, 60.0834427
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2858963, 55.2901421
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8489227, 47.8409615
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9179802, 45.9267273
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5706520, 47.5707283
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2862091, 61.2983780
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5582275, 43.5738144
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1921005, 48.2062836
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0240250, 52.0261841
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3702011, 58.3844757
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2947693, 61.2963486
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2787704, 65.2878799
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1620903, 41.1655197
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2668610, 42.2715225
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7616730, 37.7502747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9948942, upper bound: 44.8720962
time: 77.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0399372, upper bound: 44.8425831
time: 53.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 133.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.8425831, upper bound: 45.0399372
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.8720962, upper bound: 44.9948942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.8605095, upper bound: 45.0040322
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.8900212, upper bound: 44.9589991
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.8617068, upper bound: 45.0053782
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.9075415, upper bound: 44.9776748
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.9776748, upper bound: 44.9075415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 133.14
Output dim: 14, lower bound: -45.0053782, upper bound: 44.8617068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.9589991, upper bound: 44.8900212
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 133.14
Output dim: 14, lower bound: -45.0040322, upper bound: 44.8605095
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 133.14
Output dim: 14, lower bound: -44.9948942, upper bound: 44.8720962
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 133.14
Output dim: 14, lower bound: -45.0399372, upper bound: 44.8425831

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3721008, 44.3773651
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4739799, 35.4779739
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5509758, 40.5507469
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6213112, 39.6291885
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6272964, 38.6315193
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8374939, 61.8332176
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9377136, 46.9329147
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0817490, 60.0876389
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2940521, 55.2888374
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8491096, 47.8597946
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9266739, 45.9176559
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5667305, 47.5653381
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2884750, 61.2730713
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5568542, 43.5357590
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1944199, 48.1763763
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0266876, 52.0246658
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3804779, 58.3654022
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3012848, 61.3013229
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2846222, 65.2747421
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1641769, 41.1605797
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2700500, 42.2650299
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7526169, 37.7648010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8338094, upper bound: 44.9954014
time: 54.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7968449, upper bound: 45.0310707
time: 49.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3728485, 44.3765717
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4745445, 35.4774017
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5447426, 40.5566368
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6239357, 39.6265488
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6301498, 38.6284828
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8400421, 61.8304863
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9369965, 46.9336395
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0853348, 60.0838966
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2939377, 55.2889061
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8491096, 47.8598022
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9224319, 45.9217224
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5666542, 47.5654182
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2842026, 61.2773438
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5528107, 43.5398064
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1907654, 48.1800308
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0249329, 52.0264053
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3741302, 58.3717346
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2976837, 61.3048096
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2805023, 65.2788467
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1602249, 41.1643333
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2660828, 42.2689247
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7557755, 37.7616425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8517261, upper bound: 44.9594584
time: 28.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8147954, upper bound: 44.9951900
time: 69.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3732300, 44.3762360
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4751396, 35.4768066
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5509758, 40.5506897
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6223030, 39.6280594
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6272964, 38.6315193
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8354034, 61.8353043
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9360352, 46.9346161
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0819550, 60.0873413
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2929688, 55.2899246
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8519783, 47.8569374
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9257889, 45.9185486
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5662270, 47.5657616
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2846756, 61.2768631
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5510864, 43.5415306
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1902618, 48.1801643
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0267181, 52.0246429
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3798676, 58.3659515
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3021393, 61.3004990
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2837982, 65.2755432
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1637955, 41.1608734
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2697449, 42.2652702
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7528000, 37.7646103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8528690, upper bound: 44.9608116
time: 54.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8159310, upper bound: 44.9966298
time: 29.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3762360, 44.3732376
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4768028, 35.4751434
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5506859, 40.5509720
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6280556, 39.6222992
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6315231, 38.6272926
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8352966, 61.8353996
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9346161, 46.9360313
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0873413, 60.0819588
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2899170, 55.2929688
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8569374, 47.8519783
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9185562, 45.9257851
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5657616, 47.5662308
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2768631, 61.2846832
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5415344, 43.5510941
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1801682, 48.1902542
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0246429, 52.0267029
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3659515, 58.3798714
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3004913, 61.3021240
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2755280, 65.2838135
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1608734, 41.1637955
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2652740, 42.2697525
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7646103, 37.7528000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9966298, upper bound: 44.8159310
time: 26.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9608116, upper bound: 44.8528690
time: 53.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3765717, 44.3728485
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4773979, 35.4745445
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5566368, 40.5447388
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6265526, 39.6239395
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6284866, 38.6301460
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8304901, 61.8400459
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9336395, 46.9369888
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0838928, 60.0853348
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2889023, 55.2939415
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8598061, 47.8491135
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9217300, 45.9224358
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5654182, 47.5666504
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2773361, 61.2842102
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5398102, 43.5528183
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1800385, 48.1907578
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0264130, 52.0249329
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3717346, 58.3741379
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3047943, 61.2976913
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2788544, 65.2805099
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1643295, 41.1602287
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2689209, 42.2660866
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7616425, 37.7557755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9951900, upper bound: 44.8147954
time: 24.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9594584, upper bound: 44.8517261
time: 23.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3773651, 44.3721008
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4779778, 35.4739799
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5507469, 40.5509720
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6291924, 39.6213150
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6315231, 38.6272964
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8332214, 61.8374901
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9329071, 46.9377136
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0876389, 60.0817490
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2888336, 55.2940521
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8597908, 47.8491211
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9176559, 45.9266777
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5653419, 47.5667267
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2730789, 61.2884750
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5357513, 43.5568619
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1763763, 48.1944122
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0246735, 52.0266876
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3654022, 58.3804703
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3013153, 61.3012772
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2747345, 65.2846146
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1605759, 41.1641731
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2650299, 42.2700539
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7647934, 37.7526169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0310707, upper bound: 44.7968449
time: 41.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9954014, upper bound: 44.8338094
time: 30.27 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 73.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.8338094, upper bound: 44.9954014
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.7968449, upper bound: 45.0310707
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.8517261, upper bound: 44.9594584
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.8147954, upper bound: 44.9951900
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.8528690, upper bound: 44.9608116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.8159310, upper bound: 44.9966298
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.9966298, upper bound: 44.8159310
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.9608116, upper bound: 44.8528690
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.9951900, upper bound: 44.8147954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.9594584, upper bound: 44.8517261
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 73.87
Output dim: 14, lower bound: -45.0310707, upper bound: 44.7968449
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 73.87
Output dim: 14, lower bound: -44.9954014, upper bound: 44.8338094

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3677292, 44.3755569
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4725189, 35.4773636
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5477486, 40.5478668
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6202049, 39.6295929
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6264954, 38.6311760
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8383789, 61.8331413
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9392700, 46.9319000
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0808449, 60.0872650
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2950211, 55.2887497
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8489685, 47.8614616
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9264145, 45.9166603
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5652580, 47.5614510
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2874374, 61.2705688
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5542450, 43.5294571
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1922684, 48.1712036
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0266724, 52.0246582
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3802643, 58.3637619
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3011093, 61.3023376
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2845001, 65.2739792
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1640358, 41.1594162
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2700157, 42.2647057
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7521973, 37.7671814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7949612, upper bound: 44.9752858
time: 26.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7508853, upper bound: 45.0299991
time: 48.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3755569, 44.3677368
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4773636, 35.4725151
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5478630, 40.5477448
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6295967, 39.6202049
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6311722, 38.6264954
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8331299, 61.8383751
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9319000, 46.9392700
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0872688, 60.0808487
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2887421, 55.2950249
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8614655, 47.8489685
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9166489, 45.9264183
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5614433, 47.5652657
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2705612, 61.2874374
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5294495, 43.5542488
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1712036, 48.1922684
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0246582, 52.0266724
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3637543, 58.3802528
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3023300, 61.3011017
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2739716, 65.2845078
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1594124, 41.1640320
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2647057, 42.2700157
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7671814, 37.7521896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0299991, upper bound: 44.7508853
time: 27.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9752858, upper bound: 44.7949612
time: 54.40 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 84.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 84.41
Output dim: 14, lower bound: -44.7949612, upper bound: 44.9752858
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 84.41
Output dim: 14, lower bound: -44.7508853, upper bound: 45.0299991
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 84.41
Output dim: 14, lower bound: -45.0299991, upper bound: 44.7508853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 84.41
Output dim: 14, lower bound: -44.9752858, upper bound: 44.7949612

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3628883, 44.3708916
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4731483, 35.4779434
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5353966, 40.5314445
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6127930, 39.6240158
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6155014, 38.6229019
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8309402, 61.8275146
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9419327, 46.9345551
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0656929, 60.0758667
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2929916, 55.2870598
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8467865, 47.8591957
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9217453, 45.9104500
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5577965, 47.5547523
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2794647, 61.2599869
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5441208, 43.5159912
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1836243, 48.1597061
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0235825, 52.0205536
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3702850, 58.3505058
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2950134, 61.2942581
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2801437, 65.2681732
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1578827, 41.1511116
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2590446, 42.2501183
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7617188, 37.7802582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7126514, upper bound: 45.0286550
time: 48.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7499684, upper bound: 44.9938380
time: 59.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3708839, 44.3628960
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4779396, 35.4731445
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5314445, 40.5353928
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6240082, 39.6127853
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6229019, 38.6154976
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8275070, 61.8309364
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9345551, 46.9419327
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0758705, 60.0656929
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2870560, 55.2929916
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8591919, 47.8467903
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9104538, 45.9217453
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5547600, 47.5577927
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2599792, 61.2794724
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5159912, 43.5441208
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1597061, 48.1836205
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0205612, 52.0235748
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3505096, 58.3702812
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2942505, 61.2950287
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2681808, 65.2801361
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1511078, 41.1578865
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2501183, 42.2590408
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7802582, 37.7617188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9938380, upper bound: 44.7499684
time: 53.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0286550, upper bound: 44.7126514
time: 53.68 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 110.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 110.12
Output dim: 14, lower bound: -44.7126514, upper bound: 45.0286550
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 110.12
Output dim: 14, lower bound: -44.7499684, upper bound: 44.9938380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 110.12
Output dim: 14, lower bound: -44.9938380, upper bound: 44.7499684
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 110.12
Output dim: 14, lower bound: -45.0286550, upper bound: 44.7126514

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3654366, 44.3727570
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4752731, 35.4793892
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5333176, 40.5290298
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6115265, 39.6231613
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6090469, 38.6179008
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8237610, 61.8218346
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9365044, 46.9302254
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0565796, 60.0689011
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2886810, 55.2836952
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8425980, 47.8540497
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9202347, 45.9086571
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5585098, 47.5558929
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2763062, 61.2560196
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5443039, 43.5161324
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1814194, 48.1572571
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0211029, 52.0172958
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3539658, 58.3297920
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2883606, 61.2854919
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2717209, 65.2574768
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1518478, 41.1426849
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2502251, 42.2385025
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7649040, 37.7845421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1780

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6855161, upper bound: 45.0263598
time: 55.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7098828, upper bound: 44.9928153
time: 55.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3727455, 44.3654480
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4793930, 35.4752693
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5290298, 40.5333176
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6231613, 39.6115265
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6178970, 38.6090431
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8218384, 61.8237724
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9302254, 46.9365082
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0688934, 60.0565834
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2836914, 55.2886810
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8540421, 47.8426018
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9086533, 45.9202423
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5558929, 47.5585136
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2560120, 61.2762985
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5161362, 43.5443077
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1572571, 48.1814194
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0172882, 52.0211029
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3297958, 58.3539658
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2854919, 61.2883606
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2574692, 65.2717133
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1426849, 41.1518478
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2385063, 42.2502213
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7845421, 37.7649040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1780

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9928153, upper bound: 44.7098828
time: 27.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0263598, upper bound: 44.6855161
time: 53.13 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 83.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 83.34
Output dim: 14, lower bound: -44.6855161, upper bound: 45.0263598
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 83.34
Output dim: 14, lower bound: -44.7098828, upper bound: 44.9928153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 83.34
Output dim: 14, lower bound: -44.9928153, upper bound: 44.7098828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 83.34
Output dim: 14, lower bound: -45.0263598, upper bound: 44.6855161

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3688889, 44.3750572
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4780197, 35.4812546
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5298805, 40.5246658
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6099205, 39.6219215
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6006851, 38.6114731
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8141022, 61.8142052
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9296112, 46.9248390
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0447693, 60.0598755
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2829132, 55.2792091
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8369637, 47.8468132
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9184456, 45.9065018
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5605698, 47.5585861
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2727051, 61.2513351
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5447807, 43.5166359
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1794739, 48.1550522
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0178986, 52.0130997
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3342972, 58.3040810
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2794495, 61.2739182
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2616959, 65.2444153
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1417847, 41.1295586
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2387009, 42.2234039
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7682343, 37.7887306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6831818, upper bound: 45.0257964
time: 97.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6848165, upper bound: 45.0246695
time: 33.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3750534, 44.3688889
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4812546, 35.4780197
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5246620, 40.5298767
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6219215, 39.6099243
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6114731, 38.6006851
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8142090, 61.8141060
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9248428, 46.9296150
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0598755, 60.0447731
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2792053, 55.2829170
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8468056, 47.8369675
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9064980, 45.9184380
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5585861, 47.5605659
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2513428, 61.2726974
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5166359, 43.5447807
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1550522, 48.1794739
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0131073, 52.0178909
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3040848, 58.3342972
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2739258, 61.2794571
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2444229, 65.2616882
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1295547, 41.1417847
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2234039, 42.2387085
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7887268, 37.7682343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0246695, upper bound: 44.6848165
time: 37.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0257964, upper bound: 44.6831818
time: 27.45 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 66.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 66.94
Output dim: 14, lower bound: -44.6831818, upper bound: 45.0257964
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 66.94
Output dim: 14, lower bound: -44.6848165, upper bound: 45.0246695
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 66.94
Output dim: 14, lower bound: -45.0246695, upper bound: 44.6848165
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 66.94
Output dim: 14, lower bound: -45.0257964, upper bound: 44.6831818

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3691711, 44.3772163
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4751854, 35.4770775
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5102234, 40.5140305
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6078033, 39.6191521
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6040764, 38.6146545
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8095016, 61.8107681
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9170532, 46.9151535
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0570145, 60.0698929
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2884674, 55.2834244
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8339386, 47.8412361
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9187508, 45.9076843
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5506516, 47.5477333
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2717896, 61.2579346
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5117950, 43.4921646
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1687088, 48.1492271
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0081100, 52.0059357
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3026886, 58.2830048
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2804718, 61.2812195
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2460480, 65.2344666
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1098099, 41.1065865
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2141876, 42.2071114
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7620659, 37.7900772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1759

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6577955, upper bound: 45.0246939
time: 48.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6828486, upper bound: 45.0030380
time: 55.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3710480, 44.3753395
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4738426, 35.4784164
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5192490, 40.5050049
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6071472, 39.6198044
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6038704, 38.6148567
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8106766, 61.8096008
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9199295, 46.9122772
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0547867, 60.0721169
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2871246, 55.2847595
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8313904, 47.8437843
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9196205, 45.9068184
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5497055, 47.5486755
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2792969, 61.2504120
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5203094, 43.4836502
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1736526, 48.1442833
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0107346, 52.0033264
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3132172, 58.2724724
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2867584, 61.2749481
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2517548, 65.2287750
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1188126, 41.0975800
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2224121, 42.1988869
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7695885, 37.7825623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1759

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6594250, upper bound: 45.0235661
time: 51.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6844839, upper bound: 45.0019237
time: 27.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3753357, 44.3710480
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4784203, 35.4738464
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5050049, 40.5192451
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6197968, 39.6071548
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6148567, 38.6038704
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8096085, 61.8106689
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9122772, 46.9199295
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0721207, 60.0547905
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2847595, 55.2871323
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8437805, 47.8313904
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9068184, 45.9196205
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5486755, 47.5497131
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2504272, 61.2792969
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4836426, 43.5203094
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1442795, 48.1736526
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0033188, 52.0107346
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2724762, 58.3132172
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2749481, 61.2867584
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2287750, 65.2517471
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0975800, 41.1188126
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1988907, 42.2224121
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7825584, 37.7695847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1759

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0019237, upper bound: 44.6844839
time: 48.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0235661, upper bound: 44.6594250
time: 45.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3772125, 44.3691711
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4770775, 35.4751854
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5140305, 40.5102234
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6191559, 39.6078033
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6146507, 38.6040726
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8107681, 61.8095016
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9151535, 46.9170532
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0698929, 60.0570145
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2834167, 55.2884674
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8412323, 47.8339386
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9076881, 45.9187546
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5477295, 47.5506554
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2579346, 61.2717819
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4921646, 43.5117912
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1492233, 48.1687088
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0059433, 52.0081253
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2830048, 58.3026886
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2812195, 61.2804871
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2344818, 65.2460480
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1065826, 41.1098099
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2071152, 42.2141914
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7900810, 37.7620697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1759

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0030380, upper bound: 44.6828486
time: 52.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0246939, upper bound: 44.6577955
time: 36.12 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 90.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -44.6577955, upper bound: 45.0246939
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -44.6828486, upper bound: 45.0030380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -44.6594250, upper bound: 45.0235661
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -44.6844839, upper bound: 45.0019237
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -45.0019237, upper bound: 44.6844839
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -45.0235661, upper bound: 44.6594250
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -45.0030380, upper bound: 44.6828486
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 90.82
Output dim: 14, lower bound: -45.0246939, upper bound: 44.6577955

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3667145, 44.3752937
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4726524, 35.4751511
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5087204, 40.5121384
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6054459, 39.6173782
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6039505, 38.6145210
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8145828, 61.8146210
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9138985, 46.9108543
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0552216, 60.0683823
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2900696, 55.2846985
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8384285, 47.8471642
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9172821, 45.9058495
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5481415, 47.5444603
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2649536, 61.2490692
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5016022, 43.4787979
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1614838, 48.1398621
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0075836, 52.0053101
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2990570, 58.2786789
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2811890, 61.2820129
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2435913, 65.2313309
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1090851, 41.1053543
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2119064, 42.2042580
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7618675, 37.7898560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6308860, upper bound: 45.0076953
time: 54.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6403301, upper bound: 44.9985275
time: 26.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3672485, 44.3747673
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4732475, 35.4745407
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5083313, 40.5124855
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6060257, 39.6167908
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6039352, 38.6145401
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8133621, 61.8158607
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9127541, 46.9119987
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0554733, 60.0681000
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2897339, 55.2850342
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8398628, 47.8457375
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9169159, 45.9062119
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5473862, 47.5452232
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2629089, 61.2511139
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4984283, 43.4819832
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1593399, 48.1420097
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0074921, 52.0053787
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2983551, 58.2792664
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2812805, 61.2819290
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2429047, 65.2319183
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1085815, 41.1058388
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2113342, 42.2047653
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7618446, 37.7898865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6559942, upper bound: 44.9859967
time: 47.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6654188, upper bound: 44.9768166
time: 54.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3685913, 44.3734169
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4713097, 35.4764862
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5177460, 40.5031166
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6047897, 39.6180267
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6037521, 38.6147232
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8157425, 61.8134537
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9167671, 46.9079781
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0530014, 60.0706062
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2887421, 55.2860336
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8358803, 47.8497124
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9181366, 45.9049835
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5472031, 47.5454063
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2724762, 61.2415466
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5101166, 43.4702835
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1664276, 48.1349182
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0101929, 52.0027008
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3095856, 58.2681465
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2874756, 61.2757339
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2492981, 65.2256317
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1180878, 41.0963516
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2201309, 42.1960335
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7693901, 37.7823372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6325285, upper bound: 45.0065605
time: 75.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6419738, upper bound: 44.9973956
time: 29.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3691177, 44.3728905
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4719200, 35.4758797
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5173569, 40.5034599
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6053848, 39.6174431
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6037369, 38.6147385
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8145218, 61.8146973
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9156380, 46.9091225
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0532532, 60.0703239
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2884064, 55.2863693
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8373146, 47.8482857
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9177856, 45.9053459
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5464401, 47.5461655
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2704315, 61.2435913
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5069427, 43.4734688
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1642838, 48.1370659
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0101013, 52.0027695
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3088837, 58.2687378
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2875366, 61.2756500
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2486115, 65.2262192
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1175842, 41.0968361
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2195587, 42.1965408
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7693596, 37.7823715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6576427, upper bound: 44.9848792
time: 51.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6670604, upper bound: 44.9756986
time: 25.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3728867, 44.3691216
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4758720, 35.4719162
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5034637, 40.5173531
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6174393, 39.6053772
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6147385, 38.6037369
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8147049, 61.8145218
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9091225, 46.9156303
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0703201, 60.0532532
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2863617, 55.2884064
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8482857, 47.8373146
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9053497, 45.9177818
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5461655, 47.5464401
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2435913, 61.2704315
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4734650, 43.5069427
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1370621, 48.1642838
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0027771, 52.0101013
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2687378, 58.3088913
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2756653, 61.2875443
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2262115, 65.2486115
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0968323, 41.1175842
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1965408, 42.2195625
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7823753, 37.7693596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9756986, upper bound: 44.6670604
time: 24.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9848792, upper bound: 44.6576427
time: 42.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3734131, 44.3685989
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4764824, 35.4713097
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5031128, 40.5177460
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6180344, 39.6047935
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6147232, 38.6037521
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8134537, 61.8157425
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9079781, 46.9167747
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0706100, 60.0529976
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2860260, 55.2887421
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8497047, 47.8358879
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9049835, 45.9181480
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5454025, 47.5472031
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2415466, 61.2724762
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4702759, 43.5101166
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1349182, 48.1664314
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0027008, 52.0101929
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2681427, 58.3095856
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2757263, 61.2874603
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2256317, 65.2492981
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0963593, 41.1180840
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1960373, 42.2201271
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7823372, 37.7693863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9973956, upper bound: 44.6419738
time: 46.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0065605, upper bound: 44.6325285
time: 33.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3747635, 44.3672485
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4745445, 35.4732552
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5124817, 40.5083313
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6167984, 39.6060295
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6145401, 38.6039391
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8158646, 61.8133545
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9120064, 46.9127541
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0681000, 60.0554733
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2850342, 55.2897415
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8457375, 47.8398628
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9062042, 45.9169197
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5452194, 47.5473862
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2511139, 61.2629166
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4819794, 43.4984283
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1420059, 48.1593399
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0053711, 52.0074921
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2792664, 58.2983627
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2819214, 61.2812729
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2319183, 65.2429047
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1058350, 41.1085777
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2047653, 42.2113380
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7898827, 37.7618446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9768166, upper bound: 44.6654188
time: 45.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9859967, upper bound: 44.6559942
time: 26.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3752899, 44.3667221
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4751549, 35.4726486
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5121384, 40.5087242
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6173782, 39.6054459
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6145248, 38.6039581
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8146133, 61.8145790
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9108620, 46.9138947
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0683823, 60.0552216
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2846985, 55.2900772
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8471565, 47.8384361
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9058380, 45.9172821
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5444641, 47.5481491
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2490692, 61.2649536
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4788055, 43.5016022
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1398621, 48.1614876
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0053101, 52.0075836
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2786713, 58.2990570
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2820129, 61.2811813
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2313385, 65.2435913
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1053619, 41.1090813
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2042618, 42.2119026
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7898521, 37.7618713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1565

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9985275, upper bound: 44.6403301
time: 54.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0076953, upper bound: 44.6308860
time: 47.28 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 103.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6308860, upper bound: 45.0076953
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6403301, upper bound: 44.9985275
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6559942, upper bound: 44.9859967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6654188, upper bound: 44.9768166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6325285, upper bound: 45.0065605
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6419738, upper bound: 44.9973956
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6576427, upper bound: 44.9848792
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.6670604, upper bound: 44.9756986
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.9756986, upper bound: 44.6670604
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.9848792, upper bound: 44.6576427
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.9973956, upper bound: 44.6419738
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 103.87
Output dim: 14, lower bound: -45.0065605, upper bound: 44.6325285
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.9768166, upper bound: 44.6654188
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.9859967, upper bound: 44.6559942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 103.87
Output dim: 14, lower bound: -44.9985275, upper bound: 44.6403301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 103.87
Output dim: 14, lower bound: -45.0076953, upper bound: 44.6308860

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3689880, 44.3721542
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4746475, 35.4750519
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5092850, 40.5118866
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6068802, 39.6173058
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6038589, 38.6144905
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8138504, 61.8147888
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9132729, 46.9126358
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0541763, 60.0694237
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2877350, 55.2869377
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8390732, 47.8462181
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9158325, 45.9078217
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5481033, 47.5440903
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2647934, 61.2488708
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5015030, 43.4802361
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1614456, 48.1405258
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0082855, 52.0042572
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3030853, 58.2726593
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2835083, 61.2782059
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2452316, 65.2288818
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1118355, 41.1030693
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2144623, 42.2002182
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7636795, 37.7872963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6144568, upper bound: 45.0073119
time: 26.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6305018, upper bound: 44.9917090
time: 38.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3708649, 44.3702774
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4733047, 35.4763908
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5183105, 40.5028648
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6062393, 39.6179581
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6036530, 38.6146927
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8150253, 61.8136215
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9161491, 46.9097595
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0519485, 60.0716476
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2863922, 55.2882729
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8365250, 47.8487663
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9167023, 45.9069557
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5471573, 47.5450325
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2723007, 61.2413483
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5100174, 43.4717216
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1663895, 48.1355820
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0108948, 52.0016479
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3136139, 58.2621269
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2897797, 61.2719269
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2509384, 65.2231750
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1208382, 41.0940666
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2226868, 42.1919937
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7711945, 37.7797813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.6161238, upper bound: 45.0061764
time: 32.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6321539, upper bound: 44.9905759
time: 26.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3702698, 44.3685989
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4763870, 35.4713097
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5028687, 40.5177460
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6179581, 39.6047935
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6147232, 38.6036530
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8134537, 61.8150253
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9079781, 46.9161530
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0706100, 60.0519485
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2860260, 55.2863960
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8487625, 47.8358879
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9049835, 45.9166985
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5454025, 47.5471611
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2413559, 61.2724762
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4702759, 43.5100174
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1349182, 48.1663933
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0016479, 52.0101929
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2621307, 58.3095856
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2719269, 61.2874603
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2231827, 65.2492981
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0940590, 41.1180840
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1919937, 42.2201271
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7797775, 37.7693863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9905759, upper bound: 44.6321539
time: 63.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0061764, upper bound: 44.6161238
time: 75.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3721466, 44.3667221
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4750595, 35.4726486
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5118866, 40.5087242
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6173019, 39.6054459
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6145248, 38.6038551
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8146133, 61.8138580
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9108620, 46.9132729
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0683823, 60.0541725
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2846985, 55.2877312
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8462143, 47.8384361
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9058380, 45.9158325
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5444641, 47.5481071
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2488632, 61.2649536
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4788055, 43.5015030
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1398621, 48.1614494
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0042572, 52.0075836
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2726593, 58.2990570
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2782135, 61.2811813
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2288742, 65.2435913
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1030617, 41.1090813
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2002182, 42.2119026
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7872925, 37.7618713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9917090, upper bound: 44.6305019
time: 26.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0073119, upper bound: 44.6144568
time: 24.16 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 53.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 13, time: 53.22
Output dim: 14, lower bound: -44.6144568, upper bound: 45.0073119
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 53.22
Output dim: 14, lower bound: -44.6305018, upper bound: 44.9917090
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 13, time: 53.22
Output dim: 14, lower bound: -44.6161238, upper bound: 45.0061764
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 53.22
Output dim: 14, lower bound: -44.6321539, upper bound: 44.9905759
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 53.22
Output dim: 14, lower bound: -44.9905759, upper bound: 44.6321539
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 13, time: 53.22
Output dim: 14, lower bound: -45.0061764, upper bound: 44.6161238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 53.22
Output dim: 14, lower bound: -44.9917090, upper bound: 44.6305019
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 13, time: 53.22
Output dim: 14, lower bound: -45.0073119, upper bound: 44.6144568

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3539925, 44.3526421
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4799805, 35.4792480
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4991570, 40.5029678
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6156349, 39.6241455
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6008911, 38.6105499
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8167419, 61.8163033
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9127922, 46.9145317
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0518036, 60.0674095
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2756348, 55.2772713
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8414536, 47.8485641
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8997536, 45.8952179
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5487061, 47.5443764
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2611542, 61.2440567
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5016022, 43.4803123
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1657333, 48.1449509
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0063629, 52.0020294
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2799835, 58.2419205
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2795258, 61.2729187
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2336426, 65.2145615
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0939255, 41.0813179
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1994247, 42.1820984
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7401428, 37.7585220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6121169, upper bound: 44.9472823
time: 25.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5946635, upper bound: 45.0064226
time: 25.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3558693, 44.3507652
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4786377, 35.4805870
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5081749, 40.4939423
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6149864, 39.6247978
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6006851, 38.6107521
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8179169, 61.8151398
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9156685, 46.9116554
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0495834, 60.0696335
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2743073, 55.2786064
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8389053, 47.8511124
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9006233, 45.8943520
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5477676, 47.5453186
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2686615, 61.2365417
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5101166, 43.4717979
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1706772, 48.1400070
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0089722, 51.9994125
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2905273, 58.2313919
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2857971, 61.2666473
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2393494, 65.2088547
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1029282, 41.0723114
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2076492, 42.1738739
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7476501, 37.7510071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6137885, upper bound: 44.9461331
time: 22.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5963475, upper bound: 45.0052835
time: 52.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3507652, 44.3535995
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4805908, 35.4766464
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4939384, 40.5076218
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6247978, 39.6135406
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6107788, 38.6006851
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8149567, 61.8179092
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9098625, 46.9156723
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0685883, 60.0495796
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2763672, 55.2743111
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8511124, 47.8382645
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8923836, 45.9006157
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5457001, 47.5477638
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2365570, 61.2688446
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4703751, 43.5101204
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1393509, 48.1706734
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9994202, 52.0082703
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2313843, 58.2864952
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2666473, 61.2834854
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2088623, 65.2377243
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0723190, 41.1001701
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1738739, 42.2050858
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7510071, 37.7458496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0052835, upper bound: 44.5963476
time: 23.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9461331, upper bound: 44.6137885
time: 25.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3526421, 44.3517227
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4792480, 35.4779854
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5029716, 40.4985962
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6241493, 39.6141930
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6105728, 38.6008911
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8161163, 61.8167419
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9127388, 46.9127922
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0663681, 60.0518036
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2750244, 55.2756462
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8485641, 47.8408165
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8932381, 45.8997498
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5447540, 47.5487061
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2440643, 61.2613220
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4788895, 43.5016060
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1442947, 48.1657295
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0020294, 52.0056610
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2419128, 58.2759666
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2729187, 61.2772064
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2145691, 65.2320175
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0813217, 41.0911674
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1820984, 42.1968613
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7585220, 37.7383347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0064226, upper bound: 44.5946635
time: 51.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9472823, upper bound: 44.6121169
time: 28.53 seconds

## Summary of splitting (split count: 13)
- Time for RS candidates: 82.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 14, time: 82.08
Output dim: 14, lower bound: -44.6121169, upper bound: 44.9472823
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 14, time: 82.08
Output dim: 14, lower bound: -44.5946635, upper bound: 45.0064226
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 14, time: 82.08
Output dim: 14, lower bound: -44.6137885, upper bound: 44.9461331
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 14, time: 82.08
Output dim: 14, lower bound: -44.5963475, upper bound: 45.0052835
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 14, time: 82.08
Output dim: 14, lower bound: -45.0052835, upper bound: 44.5963476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 14, time: 82.08
Output dim: 14, lower bound: -44.9461331, upper bound: 44.6137885
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 14, time: 82.08
Output dim: 14, lower bound: -45.0064226, upper bound: 44.5946635
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 14, time: 82.08
Output dim: 14, lower bound: -44.9472823, upper bound: 44.6121169

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3494377, 44.3486061
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4818344, 35.4810486
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4839706, 40.4830437
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6117554, 39.6212311
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5923691, 38.6040916
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8082733, 61.8096695
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9164848, 46.9178467
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0402756, 60.0586853
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2721176, 55.2743607
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8359909, 47.8426971
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8983345, 45.8933296
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5379639, 47.5326767
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2592926, 61.2415848
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4981384, 43.4756927
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1638184, 48.1424065
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0042801, 51.9992447
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2810059, 58.2426720
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2742767, 61.2660294
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2373428, 65.2172546
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0914993, 41.0780602
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1916199, 42.1718140
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7466507, 37.7669067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5723740, upper bound: 45.0059986
time: 21.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5941936, upper bound: 44.9849117
time: 22.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3513145, 44.3467331
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4804993, 35.4823914
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4930038, 40.4740181
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6110992, 39.6218834
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5921707, 38.6042976
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8094330, 61.8085060
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9193535, 46.9149666
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0380630, 60.0609055
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2707901, 55.2756920
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8334351, 47.8452454
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8991966, 45.8924637
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5370178, 47.5336227
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2668152, 61.2340698
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5066528, 43.4671783
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1687622, 48.1374626
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0068893, 51.9966354
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2915497, 58.2321434
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2805634, 61.2597504
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2430344, 65.2115555
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1005020, 41.0690536
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1998444, 42.1635895
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7541580, 37.7593918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5740479, upper bound: 45.0048611
time: 21.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5958833, upper bound: 44.9837925
time: 24.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3467293, 44.3490372
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4823914, 35.4785156
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4740219, 40.4924393
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6218872, 39.6096687
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6043167, 38.5921707
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8083344, 61.8094444
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9131889, 46.9193535
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0598679, 60.0380592
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2734375, 55.2707901
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8452454, 47.8328094
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8904991, 45.8991966
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5339890, 47.5370216
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2340698, 61.2669830
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4657516, 43.5066528
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1368027, 48.1687622
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9966354, 52.0061951
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2321472, 58.2875290
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2597504, 61.2782440
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2115555, 65.2413864
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0690536, 41.0977516
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1635818, 42.1972885
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7593918, 37.7523499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9837925, upper bound: 44.5958833
time: 23.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0048611, upper bound: 44.5740479
time: 26.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3485985, 44.3471603
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4810486, 35.4798546
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4830399, 40.4834137
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6212311, 39.6103210
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6041183, 38.5923729
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8094940, 61.8082771
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9160576, 46.9164772
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0576477, 60.0402832
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2721100, 55.2721252
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8426971, 47.8353577
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8913536, 45.8983307
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5330582, 47.5379639
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2415771, 61.2594604
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4742661, 43.4981346
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1417465, 48.1638184
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9992447, 52.0035858
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2426758, 58.2770004
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2660370, 61.2719727
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2172623, 65.2356873
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0780563, 41.0887451
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1718063, 42.1890640
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7669144, 37.7448349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9849117, upper bound: 44.5941936
time: 46.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0059986, upper bound: 44.5723740
time: 28.36 seconds

## Summary of splitting (split count: 14)
- Time for RS candidates: 77.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 15, time: 77.31
Output dim: 14, lower bound: -44.5723740, upper bound: 45.0059986
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 15, time: 77.31
Output dim: 14, lower bound: -44.5941936, upper bound: 44.9849117
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 15, time: 77.31
Output dim: 14, lower bound: -44.5740479, upper bound: 45.0048611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 15, time: 77.31
Output dim: 14, lower bound: -44.5958833, upper bound: 44.9837925
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 15, time: 77.31
Output dim: 14, lower bound: -44.9837925, upper bound: 44.5958833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 15, time: 77.31
Output dim: 14, lower bound: -45.0048611, upper bound: 44.5740479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 15, time: 77.31
Output dim: 14, lower bound: -44.9849117, upper bound: 44.5941936
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 15, time: 77.31
Output dim: 14, lower bound: -45.0059986, upper bound: 44.5723740

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3571053, 44.3530197
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4848709, 35.4835129
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4783859, 40.4758415
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6166725, 39.6252174
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5924454, 38.6032639
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8017578, 61.8046265
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9219818, 46.9246941
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0455666, 60.0629005
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2684059, 55.2727737
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8416595, 47.8474007
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9009628, 45.8967323
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5506516, 47.5482635
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2675018, 61.2519455
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5147018, 43.4975357
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1726913, 48.1534462
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0010986, 51.9950027
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2701797, 58.2287140
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2604828, 61.2476425
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2314987, 65.2101135
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0895576, 41.0757179
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1880226, 42.1669579
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7463646, 37.7639503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5491401, upper bound: 45.0038765
time: 24.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5710716, upper bound: 44.9831534
time: 45.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3589821, 44.3511467
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4835281, 35.4848518
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4874039, 40.4668198
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6160240, 39.6258698
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5922318, 38.6034660
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8029327, 61.8034592
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9248657, 46.9218178
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0433388, 60.0651207
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2670784, 55.2741089
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8391113, 47.8499489
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9018173, 45.8958664
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5497055, 47.5492058
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2750092, 61.2444229
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5232162, 43.4890213
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1776352, 48.1484985
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0036926, 51.9923935
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2807236, 58.2181854
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2667542, 61.2413635
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2372055, 65.2044144
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0985603, 41.0667114
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1962471, 42.1587334
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7538795, 37.7564354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5508423, upper bound: 45.0027486
time: 48.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5727502, upper bound: 44.9820231
time: 56.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3511391, 44.3567085
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4848557, 35.4815407
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4668198, 40.4868469
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6258659, 39.6145897
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6034927, 38.5922394
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8032837, 61.8029289
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9200287, 46.9248619
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0640755, 60.0433426
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2718620, 55.2670784
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8499451, 47.8384705
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8938904, 45.9018250
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5495758, 47.5497055
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2444305, 61.2751846
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4875946, 43.5232162
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1478500, 48.1776352
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9924011, 52.0030136
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2181854, 58.2766800
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2413635, 61.2644272
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2044144, 65.2355499
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0667000, 41.0958176
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1587410, 42.1936836
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7564354, 37.7520714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9820231, upper bound: 44.5727502
time: 25.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0027486, upper bound: 44.5508422
time: 31.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3530159, 44.3548355
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4835129, 35.4828796
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4758377, 40.4778214
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6252174, 39.6152420
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6032791, 38.5924416
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8044586, 61.8017616
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9229126, 46.9219818
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0618553, 60.0455666
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2705269, 55.2684135
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8473969, 47.8410187
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8947601, 45.9009590
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5486298, 47.5506477
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2519379, 61.2676620
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4961090, 43.5147018
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1527939, 48.1726913
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9949951, 52.0004044
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2287140, 58.2661514
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2476349, 61.2581558
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2101212, 65.2298508
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0757179, 41.0868111
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1669655, 42.1854591
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7639503, 37.7445564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9831534, upper bound: 44.5710716
time: 64.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0038765, upper bound: 44.5491401
time: 60.64 seconds

## Summary of splitting (split count: 15)
- Time for RS candidates: 127.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 16, time: 127.91
Output dim: 14, lower bound: -44.5491401, upper bound: 45.0038765
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 16, time: 127.91
Output dim: 14, lower bound: -44.5710716, upper bound: 44.9831534
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 16, time: 127.91
Output dim: 14, lower bound: -44.5508423, upper bound: 45.0027486
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 16, time: 127.91
Output dim: 14, lower bound: -44.5727502, upper bound: 44.9820231
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 16, time: 127.91
Output dim: 14, lower bound: -44.9820231, upper bound: 44.5727502
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 16, time: 127.91
Output dim: 14, lower bound: -45.0027486, upper bound: 44.5508422
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 16, time: 127.91
Output dim: 14, lower bound: -44.9831534, upper bound: 44.5710716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 16, time: 127.91
Output dim: 14, lower bound: -45.0038765, upper bound: 44.5491401

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3552475, 44.3469353
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4861565, 35.4833527
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4766006, 40.4741592
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6181335, 39.6250381
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5921364, 38.6025314
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8018951, 61.8045692
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9207230, 46.9255371
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0444450, 60.0625191
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2638817, 55.2711983
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8419266, 47.8470383
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8964844, 45.8952827
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5502548, 47.5477676
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2671432, 61.2511749
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5145798, 43.4978027
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1726074, 48.1539383
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0006638, 51.9937363
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2668686, 58.2179222
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2588272, 61.2427826
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2300034, 65.2053452
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0878983, 41.0701904
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1859970, 42.1609001
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7438126, 37.7574615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2032
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5466523, upper bound: 45.0030572
time: 46.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5484061, upper bound: 45.0010464
time: 38.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3571091, 44.3450623
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4848213, 35.4846878
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4856262, 40.4651337
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6174850, 39.6256866
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5919304, 38.6027374
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8030548, 61.8034019
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9236069, 46.9226608
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0422325, 60.0647430
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2625542, 55.2725334
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8393784, 47.8495865
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8973541, 45.8944168
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5493164, 47.5487099
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2746506, 61.2436600
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5230942, 43.4892883
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1775513, 48.1489906
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0032883, 51.9911270
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2773972, 58.2073936
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2651138, 61.2365112
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2356949, 65.1996384
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0969009, 41.0611839
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1942215, 42.1526756
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7513351, 37.7499466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2032
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.5483732, upper bound: 45.0019301
time: 53.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5501090, upper bound: 44.9998992
time: 23.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3450546, 44.3548431
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4846916, 35.4828644
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4651337, 40.4850693
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6256866, 39.6160507
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6027565, 38.5919342
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8032379, 61.8030586
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9208755, 46.9236031
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0637245, 60.0422249
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2703743, 55.2625542
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8495865, 47.8387718
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8924408, 45.8973541
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5490799, 47.5493126
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2436447, 61.2748184
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4878082, 43.5230942
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1483307, 48.1775513
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9911270, 52.0026016
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2073898, 58.2733765
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2365189, 61.2628479
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.1996384, 65.2340546
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0611801, 41.0941544
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1526718, 42.1917191
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7499466, 37.7496147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2032
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9998992, upper bound: 44.5501090
time: 26.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0019301, upper bound: 44.5483733
time: 25.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3469315, 44.3529701
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4833488, 35.4842033
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4741592, 40.4760437
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6250381, 39.6167030
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6025505, 38.5921364
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8043976, 61.8018951
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9237518, 46.9207230
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0615044, 60.0444489
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2690468, 55.2638893
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8470383, 47.8413200
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8933105, 45.8964882
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5481415, 47.5502586
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2511826, 61.2672958
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4963226, 43.5145798
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1532745, 48.1726074
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9937363, 51.9999924
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2179184, 58.2628479
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2427750, 61.2565689
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2053452, 65.2283554
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0701828, 41.0851517
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1608963, 42.1834984
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7574615, 37.7420959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2032
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -45.0010464, upper bound: 44.5484061
time: 72.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0030572, upper bound: 44.5466523
time: 53.88 seconds

## Summary of splitting (split count: 16)
- Time for RS candidates: 128.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 17, time: 128.54
Output dim: 14, lower bound: -44.5466523, upper bound: 45.0030572
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 17, time: 128.54
Output dim: 14, lower bound: -44.5484061, upper bound: 45.0010464
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 17, time: 128.54
Output dim: 14, lower bound: -44.5483732, upper bound: 45.0019301
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 17, time: 128.54
Output dim: 14, lower bound: -44.5501090, upper bound: 44.9998992
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 17, time: 128.54
Output dim: 14, lower bound: -44.9998992, upper bound: 44.5501090
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 17, time: 128.54
Output dim: 14, lower bound: -45.0019301, upper bound: 44.5483733
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 17, time: 128.54
Output dim: 14, lower bound: -45.0010464, upper bound: 44.5484061
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 17, time: 128.54
Output dim: 14, lower bound: -45.0030572, upper bound: 44.5466523

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3555717, 44.3475685
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4801102, 35.4753189
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5007172, 40.4940491
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6118011, 39.6167297
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5967979, 38.6091805
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7977295, 61.8040886
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9167786, 46.9225311
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0411835, 60.0603867
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2581177, 55.2673607
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8461990, 47.8454514
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8875961, 45.8885880
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5541840, 47.5454025
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2708130, 61.2583389
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5018921, 43.4889030
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1470642, 48.1349297
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9997253, 51.9930420
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2427902, 58.2003593
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2583313, 61.2423401
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2427292, 65.2237396
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0840302, 41.0657883
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1925278, 42.1670113
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7586060, 37.7699661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2031
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 749

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5399215, upper bound: 44.9525373
time: 60.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5097837, upper bound: 44.9964829
time: 52.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3574486, 44.3456917
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4787674, 35.4766579
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5097351, 40.4850273
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6111603, 39.6173782
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.5965996, 38.6093826
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7989044, 61.8029213
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9196625, 46.9196548
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0389709, 60.0626068
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2567902, 55.2686958
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8436508, 47.8479996
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8884583, 45.8877220
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5532379, 47.5463486
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2783356, 61.2508240
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5104065, 43.4803886
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1520081, 48.1299820
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0023346, 51.9904327
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2533188, 58.1898308
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2646027, 61.2360611
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2484055, 65.2180481
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0930405, 41.0567818
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2007523, 42.1587868
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7661285, 37.7624512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2031
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 749

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5416296, upper bound: 44.9509890
time: 54.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5115077, upper bound: 44.9951185
time: 32.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3456841, 44.3551903
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4766617, 35.4768143
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4850311, 40.5091782
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6173859, 39.6097183
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6094170, 38.5965996
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8027496, 61.7989082
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9178772, 46.9196587
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0615921, 60.0389633
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2665253, 55.2567863
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8479919, 47.8430367
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8857498, 45.8884583
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5467148, 47.5532417
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2508240, 61.2784958
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4789124, 43.5104027
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1293259, 48.1520081
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9904327, 52.0016556
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.1898270, 58.2493057
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2360535, 61.2623444
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2180405, 65.2467880
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0567856, 41.0902824
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1587906, 42.1982498
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7624512, 37.7644081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2031
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9951185, upper bound: 44.5115077
time: 58.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9509890, upper bound: 44.5416296
time: 24.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3475609, 44.3533173
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4753189, 35.4781570
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4940491, 40.5001564
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6167297, 39.6103668
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6092110, 38.5968018
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8039246, 61.7977448
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9207458, 46.9167824
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0593643, 60.0411873
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2651978, 55.2581215
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8454437, 47.8455849
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.8866196, 45.8875961
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5457687, 47.5541878
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2583466, 61.2709732
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4874268, 43.5018883
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1342697, 48.1470642
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -51.9930420, 51.9990463
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2003555, 58.2387772
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2423401, 61.2560654
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2237473, 65.2410889
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0657883, 41.0812759
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1670151, 42.1900253
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7699661, 37.7568932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2031
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9964829, upper bound: 44.5097837
time: 51.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9525373, upper bound: 44.5399215
time: 43.96 seconds

## Summary of splitting (split count: 17)
- Time for RS candidates: 98.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.5399215, upper bound: 44.9525373
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.5097837, upper bound: 44.9964829
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.5416296, upper bound: 44.9509890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.5115077, upper bound: 44.9951185
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.9951185, upper bound: 44.5115077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.9509890, upper bound: 44.5416296
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.9964829, upper bound: 44.5097837
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 18, time: 98.46
Output dim: 14, lower bound: -44.9525373, upper bound: 44.5399215

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 63.43 + 6091.17 = 6154.61 seconds
