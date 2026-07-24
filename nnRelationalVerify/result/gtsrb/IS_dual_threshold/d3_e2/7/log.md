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
execution time: IAR + RelationalAnalysis = 3.02 + 61.16 = 64.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -45.0464004, upper bound: 45.0464004

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9148553, upper bound: 45.0457410
time: 28.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0457409, upper bound: 45.0457410
time: 30.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 59.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 59.62
Output dim: 14, lower bound: -44.9148553, upper bound: 45.0457410
IS_A2, status: Status.UNKNOWN, split count: 1, time: 59.62
Output dim: 14, lower bound: -45.0457409, upper bound: 45.0457410

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -28.0870781, 16.9735088, -28.1304779, 17.0135002, -44.3257179, 44.3286743
1: -13.6502571, 16.9623890, -13.6860161, 17.0163155, -30.6665726, 30.6484051
2: -14.0431147, 21.5172615, -14.0958071, 21.5830154, -35.3992996, 35.3843536
3: -12.8737392, 23.3187218, -12.9140253, 23.3737049, -36.2474442, 36.2327461
4: -21.5192604, 18.3360252, -21.5779648, 18.4095745, -39.9288330, 39.9139900
5: -11.9797363, 22.7273369, -12.0214148, 22.7754307, -34.7551651, 34.7487526
6: -50.6472015, -3.5886245, -50.6721230, -3.5615063, -40.5211334, 40.5074120
7: -16.3636131, 18.3187466, -16.4104958, 18.3811169, -34.7447281, 34.7292404
8: -18.2865295, 21.1883354, -18.3410950, 21.2522087, -39.5387383, 39.5294304
9: -16.6341820, 23.1681786, -16.7063293, 23.2242470, -38.5268478, 38.5465317
10: -24.1781158, 38.3754387, -24.2772408, 38.4600220, -61.6614685, 61.6826134
11: -24.6989212, 17.5711441, -24.7458096, 17.6076145, -42.3065338, 42.3169556
12: -28.4912262, 20.0362244, -28.5966797, 20.1226273, -46.7672424, 46.7883644
13: -32.8838921, 28.7340813, -32.9288864, 28.7685852, -61.6524773, 61.6629677
14: -23.3678474, 39.1069794, -23.4842453, 39.1615524, -59.9283981, 59.9922180
15: -18.9058971, 25.7981911, -18.9510117, 25.8333015, -44.7391968, 44.7492027
16: -32.6841583, 19.8102379, -32.7300262, 19.8576012, -52.5417595, 52.5402641
17: -17.6864357, 38.3958893, -17.7583504, 38.4316673, -55.1597366, 55.1971436
18: -25.7115440, 19.5758820, -25.7715588, 19.6123505, -45.3238945, 45.3474426
19: -26.3596020, 12.4963894, -26.3946457, 12.5125570, -38.8721581, 38.8910370
20: -21.0154839, 20.4286613, -21.0633602, 20.4571762, -41.4726601, 41.4920197
21: -25.6157360, 18.8764000, -25.6682415, 18.9089642, -44.5247002, 44.5446396
22: -22.0379543, 24.5219078, -22.0811577, 24.5415592, -46.5795135, 46.6030655
23: -21.6543045, 17.4881630, -21.6819496, 17.5059853, -39.1602898, 39.1701126
24: -32.0745506, 11.8690538, -32.1112938, 11.9016371, -43.9761887, 43.9803467
25: -18.0523491, 25.4135971, -18.0866985, 25.4376068, -43.4899559, 43.5002975
26: -29.1408386, 26.9209385, -29.2054596, 26.9760170, -56.1168556, 56.1263962
27: -32.0405846, 16.5158119, -32.0897217, 16.5480404, -47.7524185, 47.7720108
28: -21.4726315, 21.6962013, -21.5081635, 21.7175922, -43.1902237, 43.2043648
29: -23.6158695, 22.2008362, -23.6668091, 22.2375774, -45.8534470, 45.8676453
30: -29.5551682, 16.8327332, -29.5955448, 16.8677979, -45.8832626, 45.8928833
31: -26.2942677, 19.0959682, -26.3317184, 19.1095619, -45.4038315, 45.4276886
32: -42.1460342, 8.4606867, -42.1957550, 8.4978867, -47.4921875, 47.4968987
33: -72.2898254, -5.6492710, -72.3242722, -5.5977364, -61.2696686, 61.2507095
34: -56.4327736, -5.4849310, -56.4564171, -5.4638386, -43.5755310, 43.5829659
35: -50.0849609, 0.0440779, -50.1078568, 0.0668697, -48.1972351, 48.2102013
36: -47.6958961, 4.9686203, -47.7333031, 4.9772596, -51.9687424, 51.9957809
37: -83.5991211, -17.4577560, -83.6286011, -17.4379349, -58.3687592, 58.3549690
38: -58.5524750, 3.2490149, -58.5984497, 3.2668438, -61.2402039, 61.2587051
39: -78.8879929, -11.5791512, -78.9193497, -11.5622959, -65.2744141, 65.2844696
40: -67.6196136, -18.3507500, -67.6380539, -18.3219719, -41.1570549, 41.1339760
41: -55.1462288, -6.8348265, -55.1651764, -6.8145266, -42.2611237, 42.2261314
42: -33.9021606, 6.7806921, -33.9362030, 6.8233843, -37.6429825, 37.6334991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8699450, upper bound: 45.0411944
time: 28.02 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9103825, upper bound: 45.0411944
time: 30.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -28.1355820, 17.0317669, -28.1365471, 17.0327492, -44.3943062, 44.3817444
1: -13.6882801, 17.0419769, -13.6886787, 17.0435619, -30.7318420, 30.7306557
2: -14.0996475, 21.6154251, -14.1000652, 21.6168919, -35.4902496, 35.4714851
3: -12.9167480, 23.3999023, -12.9170551, 23.4011879, -36.3179359, 36.3169556
4: -21.5844002, 18.4456673, -21.5849476, 18.4470043, -40.0314026, 40.0306168
5: -12.0241070, 22.7993088, -12.0243931, 22.8003006, -34.8244095, 34.8237000
6: -50.6817017, -3.5557694, -50.6831665, -3.5548744, -40.5643387, 40.5819244
7: -16.4137650, 18.4085808, -16.4142132, 18.4121304, -34.8258972, 34.8227921
8: -18.3470612, 21.2830067, -18.3476372, 21.2849350, -39.6319962, 39.6306458
9: -16.7384319, 23.2318439, -16.7397346, 23.2324486, -38.6202774, 38.6451950
10: -24.3229694, 38.4702644, -24.3246460, 38.4712296, -61.7952499, 61.8261528
11: -24.7676888, 17.6122360, -24.7688580, 17.6131287, -42.3808174, 42.3810959
12: -28.6491661, 20.1301270, -28.6513252, 20.1306877, -46.9120445, 46.9365692
13: -32.9402466, 28.7770615, -32.9475327, 28.7779465, -61.7181931, 61.7245941
14: -23.5345154, 39.1657028, -23.5378742, 39.1660614, -60.0680084, 60.1064682
15: -18.9595680, 25.8473969, -18.9610672, 25.8494072, -44.8089752, 44.8084641
16: -32.7438965, 19.8658772, -32.7480621, 19.8672600, -52.6111565, 52.6139374
17: -17.7915306, 38.4366798, -17.7936707, 38.4372787, -55.2515411, 55.2734756
18: -25.7841625, 19.6263847, -25.7856216, 19.6271248, -45.4112854, 45.4120064
19: -26.4083824, 12.5141115, -26.4098530, 12.5144043, -38.9227867, 38.9239655
20: -21.0836926, 20.4591732, -21.0847740, 20.4594383, -41.5431290, 41.5439453
21: -25.6925869, 18.9116783, -25.6938553, 18.9120159, -44.6046028, 44.6055336
22: -22.0918980, 24.5453949, -22.0998077, 24.5457497, -46.6376495, 46.6452026
23: -21.6918373, 17.5077477, -21.6936340, 17.5081291, -39.1999664, 39.2013817
24: -32.1177902, 11.9159832, -32.1185799, 11.9165268, -44.0343170, 44.0345612
25: -18.0992489, 25.4406853, -18.1023693, 25.4410477, -43.5402985, 43.5430527
26: -29.2270889, 26.9795990, -29.2335167, 26.9800034, -56.2070923, 56.2131157
27: -32.0990295, 16.5606003, -32.1000595, 16.5628853, -47.8237686, 47.7949600
28: -21.5222797, 21.7197304, -21.5235252, 21.7201118, -43.2423935, 43.2432556
29: -23.6880074, 22.2407990, -23.6913681, 22.2410946, -45.9291000, 45.9321671
30: -29.6143169, 16.8724842, -29.6154709, 16.8731956, -45.9455109, 45.9544258
31: -26.3459396, 19.1125793, -26.3470020, 19.1130676, -45.4590073, 45.4595795
32: -42.2186890, 8.5021601, -42.2199020, 8.5028391, -47.5691757, 47.5816612
33: -72.3282928, -5.5769386, -72.3289337, -5.5758896, -61.3334503, 61.3310089
34: -56.4618378, -5.4570513, -56.4623718, -5.4564142, -43.6247101, 43.6188507
35: -50.1130714, 0.0751648, -50.1138306, 0.0762978, -48.2776031, 48.2528076
36: -47.7456055, 4.9801769, -47.7504044, 4.9805441, -52.0265503, 52.0338211
37: -83.6378555, -17.4284172, -83.6395874, -17.4275780, -58.4245148, 58.4703903
38: -58.6108856, 3.2721672, -58.6120491, 3.2739906, -61.2995758, 61.2956696
39: -78.9277573, -11.5567026, -78.9285431, -11.5559940, -65.3194580, 65.3164597
40: -67.6444778, -18.3145294, -67.6455307, -18.3091927, -41.1780434, 41.2283554
41: -55.1720467, -6.8087120, -55.1725006, -6.8074694, -42.2826195, 42.3301849
42: -33.9523735, 6.8292742, -33.9529686, 6.8300667, -37.7031937, 37.7080116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8699450, upper bound: 45.0411944
time: 51.34 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0411940, upper bound: 45.0411944
time: 49.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 103.55 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 103.55
Output dim: 14, lower bound: -44.8699450, upper bound: 45.0411944
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 103.55
Output dim: 14, lower bound: -44.9103825, upper bound: 45.0411944
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 103.55
Output dim: 14, lower bound: -44.8699450, upper bound: 45.0411944
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 103.55
Output dim: 14, lower bound: -45.0411940, upper bound: 45.0411944

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -28.0290031, 16.9682693, -28.1233292, 17.0128746, -44.2692642, 44.3168488
1: -13.6168499, 16.9584541, -13.6819324, 17.0158367, -30.6326866, 30.6403866
2: -14.0098724, 21.5132790, -14.0918055, 21.5825348, -35.3663635, 35.3762207
3: -12.8445797, 23.3119583, -12.9103966, 23.3728962, -36.2174759, 36.2223549
4: -21.4886532, 18.3249741, -21.5742569, 18.4082375, -39.8968887, 39.8992310
5: -11.9429483, 22.7214489, -12.0169363, 22.7747154, -34.7176628, 34.7383842
6: -50.6431007, -3.6458564, -50.6716156, -3.5683799, -40.5102234, 40.4500198
7: -16.3269920, 18.3142242, -16.4060516, 18.3805809, -34.7075729, 34.7202759
8: -18.2241211, 21.1819286, -18.3335037, 21.2514381, -39.4755592, 39.5154343
9: -16.5959473, 23.1628914, -16.7016602, 23.2236137, -38.4876709, 38.5363541
10: -24.1350365, 38.3639221, -24.2718792, 38.4586334, -61.6138306, 61.6647797
11: -24.6922550, 17.5553379, -24.7450104, 17.6057014, -42.2979584, 42.3003464
12: -28.4862499, 19.9986076, -28.5961056, 20.1180534, -46.7525673, 46.7530556
13: -32.8691902, 28.7143059, -32.9271317, 28.7661819, -61.6353722, 61.6414375
14: -23.2940788, 39.1017570, -23.4752693, 39.1609383, -59.8536339, 59.9773865
15: -18.8643303, 25.7899551, -18.9459877, 25.8323021, -44.6966324, 44.7359428
16: -32.6546555, 19.8006039, -32.7265053, 19.8564529, -52.5111084, 52.5271072
17: -17.6440125, 38.3903732, -17.7531261, 38.4309769, -55.1138802, 55.1863060
18: -25.7035046, 19.5642281, -25.7705975, 19.6109428, -45.3144455, 45.3348236
19: -26.3503990, 12.4646931, -26.3935432, 12.5087547, -38.8591537, 38.8582382
20: -21.0057697, 20.3989334, -21.0621948, 20.4535713, -41.4593430, 41.4611282
21: -25.6032696, 18.8343391, -25.6667347, 18.9039040, -44.5071716, 44.5010757
22: -22.0284939, 24.4939861, -22.0800133, 24.5380535, -46.5665474, 46.5739975
23: -21.6439266, 17.4769440, -21.6806850, 17.5046177, -39.1485443, 39.1576309
24: -32.0668564, 11.8484678, -32.1103592, 11.8991661, -43.9660225, 43.9588280
25: -18.0402718, 25.3923740, -18.0852509, 25.4350204, -43.4752922, 43.4776230
26: -29.1295567, 26.8945980, -29.2041187, 26.9727421, -56.1022987, 56.0987167
27: -32.0308418, 16.4944077, -32.0885696, 16.5453491, -47.7390976, 47.7393723
28: -21.4634857, 21.6720238, -21.5070515, 21.7146606, -43.1781464, 43.1790771
29: -23.6097145, 22.1860447, -23.6660652, 22.2357883, -45.8455048, 45.8521118
30: -29.5465393, 16.8056583, -29.5944786, 16.8644104, -45.8714218, 45.8679237
31: -26.2788811, 19.0565643, -26.3298683, 19.1046677, -45.3835487, 45.3864326
32: -42.1400185, 8.4198875, -42.1950378, 8.4929256, -47.4815407, 47.4556351
33: -72.2794952, -5.7211990, -72.3230286, -5.6063852, -61.2494507, 61.1773987
34: -56.4264145, -5.5398731, -56.4556656, -5.4704142, -43.5621338, 43.5289421
35: -50.0770798, -0.0182180, -50.1069183, 0.0594158, -48.1813278, 48.1463699
36: -47.6878967, 4.9045296, -47.7323456, 4.9695749, -51.9528656, 51.9307480
37: -83.5898361, -17.4987049, -83.6274719, -17.4428501, -58.3534927, 58.3120117
38: -58.5389061, 3.1735191, -58.5968475, 3.2577152, -61.2169647, 61.1808395
39: -78.8759689, -11.6492004, -78.9179153, -11.5706940, -65.2541962, 65.2126465
40: -67.6108170, -18.3790436, -67.6370316, -18.3254032, -41.1441040, 41.1065102
41: -55.1422729, -6.8796673, -55.1646957, -6.8200016, -42.2508240, 42.1786270
42: -33.8971901, 6.7625809, -33.9356041, 6.8211603, -37.6357155, 37.6126060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8661090, upper bound: 44.8882777
time: 47.05 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8678867, upper bound: 45.0405589
time: 37.89 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -28.1159019, 17.0342369, -28.1271057, 17.0127831, -44.3496475, 44.3888474
1: -13.6533222, 17.0027332, -13.6842489, 17.0157070, -30.6690292, 30.6869812
2: -14.0453510, 21.5605450, -14.0935802, 21.5824928, -35.3988533, 35.4292755
3: -12.8710842, 23.3505936, -12.9111252, 23.3728523, -36.2439346, 36.2617188
4: -21.5213470, 18.3461800, -21.5749683, 18.4073906, -39.9287376, 39.9211502
5: -11.9781666, 22.7573605, -12.0185089, 22.7747383, -34.7529068, 34.7758713
6: -50.7503662, -3.5866332, -50.6713905, -3.5668292, -40.6285820, 40.5055008
7: -16.3636723, 18.3480339, -16.4076748, 18.3805809, -34.7442551, 34.7557068
8: -18.2828407, 21.2516575, -18.3363514, 21.2515945, -39.5344353, 39.5880089
9: -16.6353455, 23.2204208, -16.7018623, 23.2234650, -38.5246506, 38.5922394
10: -24.1860237, 38.4449310, -24.2729359, 38.4588242, -61.6665649, 61.7538223
11: -24.7265968, 17.5640640, -24.7447567, 17.6031723, -42.3297691, 42.3088226
12: -28.5048943, 20.0455379, -28.5959511, 20.1195946, -46.7696457, 46.7975616
13: -32.8927002, 28.7622223, -32.9263344, 28.7666435, -61.6593437, 61.6885567
14: -23.3927860, 39.1804886, -23.4773369, 39.1608658, -59.9513397, 60.0595474
15: -18.9149303, 25.8431664, -18.9468575, 25.8321533, -44.7470856, 44.7900238
16: -32.6928062, 19.8610287, -32.7263412, 19.8567104, -52.5495148, 52.5873718
17: -17.6997910, 38.4645081, -17.7546082, 38.4307938, -55.1689148, 55.2660828
18: -25.7737865, 19.5758629, -25.7708607, 19.6091118, -45.3828964, 45.3467255
19: -26.4394073, 12.4958658, -26.3935013, 12.5107098, -38.9501190, 38.8893661
20: -21.0958939, 20.4321709, -21.0622673, 20.4556122, -41.5515060, 41.4944382
21: -25.7105465, 18.8766212, -25.6668587, 18.9065132, -44.6170578, 44.5434799
22: -22.1123734, 24.5209713, -22.0802269, 24.5399017, -46.6522751, 46.6011963
23: -21.6801815, 17.4963074, -21.6801815, 17.5050125, -39.1851959, 39.1764908
24: -32.1664162, 11.8649330, -32.1105995, 11.8976870, -44.0641022, 43.9755325
25: -18.0941715, 25.4067078, -18.0855026, 25.4322681, -43.5264397, 43.4922104
26: -29.1816235, 26.9141617, -29.2042999, 26.9698410, -56.1514664, 56.1184616
27: -32.0949020, 16.5128555, -32.0887375, 16.5443687, -47.8290062, 47.7614365
28: -21.5112782, 21.6927299, -21.5071106, 21.7122726, -43.2235489, 43.1998405
29: -23.6776276, 22.1973419, -23.6660881, 22.2348671, -45.9124947, 45.8634300
30: -29.6330185, 16.8376312, -29.5944481, 16.8633728, -45.9522705, 45.8995285
31: -26.3980541, 19.0955257, -26.3299026, 19.1073437, -45.5053978, 45.4254303
32: -42.2146988, 8.4677963, -42.1949348, 8.4950123, -47.5583687, 47.5018196
33: -72.4296722, -5.6542816, -72.3232269, -5.6028433, -61.4033127, 61.2441025
34: -56.5227203, -5.4888525, -56.4556618, -5.4678326, -43.6663437, 43.5784531
35: -50.1988602, 0.0363798, -50.1069870, 0.0626221, -48.3070068, 48.2012672
36: -47.7885971, 4.9635506, -47.7325096, 4.9727650, -52.0587616, 51.9895401
37: -83.7028732, -17.4577751, -83.6272659, -17.4410019, -58.4707642, 58.3520355
38: -58.6976089, 3.2437572, -58.5971870, 3.2612286, -61.3790970, 61.2515106
39: -79.0402985, -11.5842285, -78.9177094, -11.5670500, -65.4236603, 65.2774811
40: -67.7047119, -18.3459206, -67.6369781, -18.3240738, -41.2409897, 41.1378975
41: -55.2040329, -6.8303452, -55.1643982, -6.8178740, -42.3188515, 42.2273254
42: -33.9108429, 6.7826090, -33.9353294, 6.8194227, -37.6498718, 37.6342354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9065248, upper bound: 44.8882777
time: 56.56 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9083236, upper bound: 45.0405589
time: 27.33 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -28.0774879, 17.0265369, -28.1294098, 17.0321331, -44.3378296, 44.3699341
1: -13.6548882, 17.0380478, -13.6846237, 17.0430794, -30.6979675, 30.7226715
2: -14.0664110, 21.6114540, -14.0960598, 21.6164131, -35.4573174, 35.4633675
3: -12.8875885, 23.3931332, -12.9134302, 23.4003716, -36.2879601, 36.3065643
4: -21.5537739, 18.4346085, -21.5812187, 18.4456940, -39.9994659, 40.0158272
5: -11.9873228, 22.7934151, -12.0198994, 22.7995949, -34.7869186, 34.8133163
6: -50.6775932, -3.6129541, -50.6826935, -3.5617237, -40.5534554, 40.5245056
7: -16.3771324, 18.4040680, -16.4097824, 18.4115982, -34.7887306, 34.8138504
8: -18.2846413, 21.2765999, -18.3400726, 21.2841549, -39.5687943, 39.6166725
9: -16.7002106, 23.2265606, -16.7350636, 23.2318192, -38.5811539, 38.6349945
10: -24.2798538, 38.4587326, -24.3192921, 38.4698601, -61.7476044, 61.8082924
11: -24.7610340, 17.5964146, -24.7680531, 17.6111870, -42.3722229, 42.3644676
12: -28.6441727, 20.0924530, -28.6507072, 20.1261425, -46.8973389, 46.9012489
13: -32.9255409, 28.7573071, -32.9457703, 28.7755756, -61.7011185, 61.7030792
14: -23.4607811, 39.1605110, -23.5288830, 39.1654320, -59.9932861, 60.0916595
15: -18.9179764, 25.8391685, -18.9560432, 25.8484230, -44.7663994, 44.7952118
16: -32.7143860, 19.8562412, -32.7445297, 19.8661194, -52.5805054, 52.6007690
17: -17.7491417, 38.4311371, -17.7884617, 38.4366341, -55.2057266, 55.2626343
18: -25.7761021, 19.6147499, -25.7846813, 19.6257267, -45.4018288, 45.3994293
19: -26.3991852, 12.4823990, -26.4087601, 12.5105724, -38.9097595, 38.8911591
20: -21.0739784, 20.4294510, -21.0836086, 20.4558487, -41.5298271, 41.5130615
21: -25.6801262, 18.8695793, -25.6923428, 18.9069595, -44.5870857, 44.5619202
22: -22.0824356, 24.5174503, -22.0986805, 24.5422287, -46.6246643, 46.6161308
23: -21.6814575, 17.4965172, -21.6923790, 17.5067596, -39.1882172, 39.1888962
24: -32.1101303, 11.8953838, -32.1176567, 11.9140558, -44.0241852, 44.0130386
25: -18.0871391, 25.4194756, -18.1009293, 25.4384670, -43.5256042, 43.5204048
26: -29.2157478, 26.9532471, -29.2321587, 26.9767799, -56.1925278, 56.1854057
27: -32.0893021, 16.5391788, -32.0988998, 16.5601711, -47.8104591, 47.7622948
28: -21.5131245, 21.6955395, -21.5224266, 21.7171612, -43.2302856, 43.2179642
29: -23.6818199, 22.2260551, -23.6906204, 22.2393169, -45.9211349, 45.9166756
30: -29.6056709, 16.8453979, -29.6144352, 16.8698082, -45.9336472, 45.9294624
31: -26.3305264, 19.0731888, -26.3451347, 19.1081409, -45.4386673, 45.4183235
32: -42.2126656, 8.4613991, -42.2191925, 8.4978590, -47.5585861, 47.5403938
33: -72.3179321, -5.6487923, -72.3276520, -5.5844498, -61.3132782, 61.2577667
34: -56.4554977, -5.5120068, -56.4615936, -5.4629812, -43.6113548, 43.5648346
35: -50.1052246, 0.0128880, -50.1128845, 0.0688763, -48.2616882, 48.1890068
36: -47.7376251, 4.9160633, -47.7494507, 4.9729080, -52.0106354, 51.9687729
37: -83.6285629, -17.4693642, -83.6384659, -17.4324493, -58.4092255, 58.4274521
38: -58.5973129, 3.1966991, -58.6104584, 3.2648602, -61.2762909, 61.2178192
39: -78.9156952, -11.6267395, -78.9271393, -11.5643950, -65.2992249, 65.2446289
40: -67.6357040, -18.3427792, -67.6444778, -18.3126259, -41.1650925, 41.2008896
41: -55.1681061, -6.8535395, -55.1720200, -6.8129206, -42.2723274, 42.2826843
42: -33.9474106, 6.8111572, -33.9523697, 6.8278208, -37.6959381, 37.6871338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9978264, upper bound: 44.8882777
time: 42.15 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0001410, upper bound: 45.0405589
time: 50.15 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -28.1643906, 17.0924854, -28.1331863, 17.0320454, -44.4182281, 44.4419212
1: -13.6913548, 17.0823326, -13.6869116, 17.0429420, -30.7342968, 30.7692451
2: -14.1018715, 21.6587181, -14.0978069, 21.6163712, -35.4898071, 35.5164337
3: -12.9140778, 23.4317436, -12.9141493, 23.4003334, -36.3144112, 36.3458939
4: -21.5864792, 18.4558144, -21.5819454, 18.4448395, -40.0313187, 40.0377579
5: -12.0225353, 22.8293266, -12.0214672, 22.7996140, -34.8221512, 34.8507919
6: -50.7848930, -3.5537453, -50.6824570, -3.5602069, -40.6718063, 40.5800247
7: -16.4138069, 18.4378643, -16.4113922, 18.4115982, -34.8254051, 34.8492584
8: -18.3433781, 21.3463516, -18.3428688, 21.2843189, -39.6276970, 39.6892204
9: -16.7396259, 23.2840805, -16.7352753, 23.2316647, -38.6181030, 38.6908836
10: -24.3308201, 38.5397263, -24.3203239, 38.4700546, -61.8003387, 61.8972855
11: -24.7953815, 17.6051445, -24.7678032, 17.6086693, -42.4040527, 42.3729477
12: -28.6627922, 20.1394482, -28.6505852, 20.1276608, -46.9144135, 46.9457588
13: -32.9490623, 28.8052158, -32.9449387, 28.7759800, -61.7250443, 61.7501526
14: -23.5594559, 39.2391891, -23.5309811, 39.1653938, -60.0909805, 60.1737671
15: -18.9686089, 25.8923855, -18.9569187, 25.8482704, -44.8168793, 44.8493042
16: -32.7525330, 19.9166546, -32.7443771, 19.8663788, -52.6189117, 52.6610336
17: -17.8049126, 38.5052567, -17.7899551, 38.4364319, -55.2607498, 55.3424225
18: -25.8463173, 19.6263790, -25.7849140, 19.6238670, -45.4701843, 45.4112930
19: -26.4881954, 12.5135956, -26.4087143, 12.5125160, -39.0007095, 38.9223099
20: -21.1640987, 20.4626770, -21.0836849, 20.4578819, -41.6219788, 41.5463638
21: -25.7873917, 18.9118958, -25.6924725, 18.9095917, -44.6969833, 44.6043701
22: -22.1662998, 24.5444450, -22.0988998, 24.5440674, -46.7103653, 46.6433449
23: -21.7177086, 17.5158997, -21.6918850, 17.5071335, -39.2248421, 39.2077866
24: -32.2096405, 11.9118538, -32.1178703, 11.9125824, -44.1222229, 44.0297241
25: -18.1410465, 25.4337921, -18.1011868, 25.4357109, -43.5767593, 43.5349808
26: -29.2679043, 26.9728050, -29.2323742, 26.9738579, -56.2417603, 56.2051773
27: -32.1532669, 16.5576591, -32.0990791, 16.5592060, -47.9003563, 47.7843895
28: -21.5609474, 21.7162533, -21.5224762, 21.7147617, -43.2757111, 43.2387314
29: -23.7497654, 22.2373390, -23.6906509, 22.2383728, -45.9881363, 45.9279900
30: -29.6921482, 16.8773556, -29.6143913, 16.8687649, -46.0144958, 45.9610748
31: -26.4496899, 19.1121635, -26.3451462, 19.1108475, -45.5605392, 45.4573097
32: -42.2873840, 8.5093203, -42.2190781, 8.4999409, -47.6354141, 47.5865936
33: -72.4681244, -5.5818825, -72.3278656, -5.5809336, -61.4671021, 61.3244781
34: -56.5517960, -5.4609919, -56.4616165, -5.4603863, -43.7155571, 43.6143074
35: -50.2269974, 0.0674562, -50.1129303, 0.0720997, -48.3873444, 48.2438736
36: -47.8383064, 4.9751244, -47.7495804, 4.9760447, -52.1165543, 52.0275650
37: -83.7415924, -17.4284172, -83.6382523, -17.4306355, -58.5264893, 58.4674911
38: -58.7560043, 3.2669210, -58.6107903, 3.2683678, -61.4384308, 61.2884827
39: -79.0800171, -11.5617828, -78.9269028, -11.5607080, -65.4686890, 65.3094406
40: -67.7295532, -18.3096809, -67.6444397, -18.3112755, -41.2620087, 41.2322655
41: -55.2298355, -6.8042164, -55.1717224, -6.8108044, -42.3403511, 42.3313904
42: -33.9610977, 6.8311901, -33.9520874, 6.8260956, -37.7100830, 37.7087479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0382157, upper bound: 44.8882777
time: 24.11 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0405587, upper bound: 45.0405589
time: 30.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 56.68 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 56.68
Output dim: 14, lower bound: -44.8661090, upper bound: 44.8882777
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 56.68
Output dim: 14, lower bound: -44.8678867, upper bound: 45.0405589
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 56.68
Output dim: 14, lower bound: -44.9065248, upper bound: 44.8882777
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 56.68
Output dim: 14, lower bound: -44.9083236, upper bound: 45.0405589
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 56.68
Output dim: 14, lower bound: -44.9978264, upper bound: 44.8882777
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 56.68
Output dim: 14, lower bound: -45.0001410, upper bound: 45.0405589
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 56.68
Output dim: 14, lower bound: -45.0382157, upper bound: 44.8882777
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 56.68
Output dim: 14, lower bound: -45.0405587, upper bound: 45.0405589

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -28.0288239, 16.9681683, -28.1213455, 17.0118942, -44.2566681, 44.3147392
1: -13.6167545, 16.9583168, -13.6810341, 17.0143356, -30.6310902, 30.6393509
2: -14.0097818, 21.5131435, -14.0910397, 21.5812035, -35.3518105, 35.3752747
3: -12.8445177, 23.3116169, -12.9097910, 23.3697548, -36.2142715, 36.2214088
4: -21.4885654, 18.3248711, -21.5733509, 18.4072418, -39.8958054, 39.8982239
5: -11.9428740, 22.7212563, -12.0162449, 22.7731552, -34.7160301, 34.7375031
6: -50.6429863, -3.6459570, -50.6705055, -3.5694628, -40.5231476, 40.4480095
7: -16.3268967, 18.3137321, -16.4051094, 18.3763142, -34.7032089, 34.7188416
8: -18.2240028, 21.1817474, -18.3323364, 21.2496128, -39.4736176, 39.5140839
9: -16.5957832, 23.1628265, -16.7001343, 23.2230186, -38.4867935, 38.5165443
10: -24.1348171, 38.3638382, -24.2694454, 38.4577026, -61.6124115, 61.6458511
11: -24.6921120, 17.5552387, -24.7436581, 17.6047211, -42.2968330, 42.2988968
12: -28.4860573, 19.9985428, -28.5939903, 20.1173000, -46.7515450, 46.7340202
13: -32.8690186, 28.7141590, -32.9253387, 28.7647743, -61.6337929, 61.6394958
14: -23.2936745, 39.1017265, -23.4709282, 39.1604843, -59.8527145, 59.9420242
15: -18.8638420, 25.7897720, -18.9405022, 25.8305626, -44.6944046, 44.7302742
16: -32.6544571, 19.8004646, -32.7246284, 19.8551979, -52.5096550, 52.5250931
17: -17.6437874, 38.3902893, -17.7508545, 38.4301300, -55.1127930, 55.1668167
18: -25.7033329, 19.5641670, -25.7688065, 19.6103058, -45.3136368, 45.3329735
19: -26.3502979, 12.4646549, -26.3924751, 12.5083885, -38.8586884, 38.8571320
20: -21.0056362, 20.3988991, -21.0607796, 20.4531727, -41.4588089, 41.4596786
21: -25.6030025, 18.8342915, -25.6639500, 18.9034843, -44.5064850, 44.4982414
22: -22.0275707, 24.4939365, -22.0704536, 24.5375099, -46.5650787, 46.5643921
23: -21.6438446, 17.4768143, -21.6799049, 17.5033684, -39.1472130, 39.1567192
24: -32.0667648, 11.8484068, -32.1093903, 11.8985558, -43.9653206, 43.9577980
25: -18.0400276, 25.3923187, -18.0827332, 25.4344902, -43.4745178, 43.4750519
26: -29.1285667, 26.8945045, -29.1941948, 26.9718342, -56.1004028, 56.0886993
27: -32.0306854, 16.4943619, -32.0871468, 16.5446301, -47.7144623, 47.7378464
28: -21.4633808, 21.6719627, -21.5062180, 21.7140274, -43.1774063, 43.1781807
29: -23.6091232, 22.1860085, -23.6605511, 22.2354031, -45.8445282, 45.8465576
30: -29.5464363, 16.8055840, -29.5934830, 16.8635368, -45.8702774, 45.8579254
31: -26.2787170, 19.0563087, -26.3281708, 19.1017551, -45.3804703, 45.3844795
32: -42.1398621, 8.4198189, -42.1935425, 8.4921751, -47.4971275, 47.4533272
33: -72.2794495, -5.7213583, -72.3225250, -5.6080360, -61.2354355, 61.1767273
34: -56.4263535, -5.5403013, -56.4550858, -5.4750080, -43.5536995, 43.5280952
35: -50.0770187, -0.0183678, -50.1062965, 0.0579796, -48.1792831, 48.1629105
36: -47.6874619, 4.9044447, -47.7278061, 4.9687901, -51.9569244, 51.9265518
37: -83.5897293, -17.4988022, -83.6263580, -17.4439087, -58.3501663, 58.3050079
38: -58.5387878, 3.1734362, -58.5954552, 3.2568531, -61.2255554, 61.1759415
39: -78.8758545, -11.6492577, -78.9170532, -11.5715666, -65.2460175, 65.2114944
40: -67.6106949, -18.3793163, -67.6356201, -18.3278522, -41.1665993, 41.1020203
41: -55.1421890, -6.8799667, -55.1639099, -6.8230419, -42.2859116, 42.1744156
42: -33.8971558, 6.7624979, -33.9351845, 6.8202991, -37.6538315, 37.6111565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8642075, upper bound: 44.9152451
time: 32.86 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8671733, upper bound: 45.0398400
time: 42.23 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -28.1157188, 17.0341339, -28.1251144, 17.0118065, -44.3370667, 44.3867035
1: -13.6532440, 17.0025978, -13.6833248, 17.0142059, -30.6674500, 30.6859226
2: -14.0452938, 21.5604019, -14.0928192, 21.5811501, -35.3843079, 35.4283447
3: -12.8710251, 23.3502426, -12.9105225, 23.3697128, -36.2407379, 36.2607651
4: -21.5212555, 18.3460732, -21.5740967, 18.4063988, -39.9276543, 39.9201698
5: -11.9781170, 22.7571697, -12.0178261, 22.7731876, -34.7513046, 34.7749939
6: -50.7502480, -3.5867472, -50.6702805, -3.5679054, -40.6415176, 40.5035057
7: -16.3635826, 18.3475323, -16.4067135, 18.3763199, -34.7399025, 34.7542458
8: -18.2827263, 21.2514801, -18.3351631, 21.2497501, -39.5324783, 39.5866432
9: -16.6351929, 23.2203636, -16.7003326, 23.2228775, -38.5237732, 38.5724335
10: -24.1857796, 38.4448433, -24.2705116, 38.4578781, -61.6651306, 61.7348671
11: -24.7264690, 17.5639744, -24.7433777, 17.6021919, -42.3286591, 42.3073502
12: -28.5047112, 20.0454674, -28.5938320, 20.1188316, -46.7686386, 46.7784882
13: -32.8925476, 28.7620850, -32.9244881, 28.7652283, -61.6577759, 61.6865730
14: -23.3924007, 39.1804352, -23.4730511, 39.1604195, -59.9504700, 60.0241890
15: -18.9144115, 25.8429947, -18.9413605, 25.8303986, -44.7448120, 44.7843552
16: -32.6926308, 19.8609200, -32.7244835, 19.8554649, -52.5480957, 52.5854034
17: -17.6995888, 38.4644241, -17.7523232, 38.4299164, -55.1678505, 55.2466240
18: -25.7735901, 19.5758018, -25.7690372, 19.6084824, -45.3820724, 45.3448410
19: -26.4392967, 12.4958410, -26.3924446, 12.5103168, -38.9496155, 38.8882866
20: -21.0957470, 20.4321346, -21.0608673, 20.4551907, -41.5509377, 41.4930038
21: -25.7102699, 18.8765774, -25.6640701, 18.9061089, -44.6163788, 44.5406494
22: -22.1114731, 24.5209122, -22.0706787, 24.5393715, -46.6508446, 46.5915909
23: -21.6800938, 17.4961967, -21.6793861, 17.5037498, -39.1838455, 39.1755829
24: -32.1663284, 11.8648777, -32.1096153, 11.8970985, -44.0634270, 43.9744949
25: -18.0939465, 25.4066525, -18.0829945, 25.4317284, -43.5256729, 43.4896469
26: -29.1806374, 26.9140987, -29.1943989, 26.9689407, -56.1495781, 56.1084976
27: -32.0947647, 16.5127811, -32.0873184, 16.5436592, -47.8043861, 47.7599144
28: -21.5111923, 21.6926689, -21.5062580, 21.7116127, -43.2228050, 43.1989288
29: -23.6770630, 22.1973152, -23.6605682, 22.2344589, -45.9115219, 45.8578835
30: -29.6329193, 16.8375473, -29.5934486, 16.8624992, -45.9511414, 45.8895416
31: -26.3978996, 19.0952702, -26.3281975, 19.1044369, -45.5023346, 45.4234695
32: -42.2145576, 8.4677372, -42.1934509, 8.4942617, -47.5739822, 47.4995155
33: -72.4296188, -5.6544580, -72.3227081, -5.6045551, -61.3893509, 61.2434311
34: -56.5226555, -5.4892807, -56.4550972, -5.4723873, -43.6579208, 43.5775833
35: -50.1987991, 0.0362473, -50.1063347, 0.0611935, -48.3049469, 48.2178078
36: -47.7881699, 4.9634743, -47.7279358, 4.9719791, -52.0628052, 51.9853516
37: -83.7027588, -17.4578552, -83.6260910, -17.4420433, -58.4674377, 58.3450470
38: -58.6974678, 3.2436829, -58.5958176, 3.2603731, -61.3876801, 61.2465744
39: -79.0402374, -11.5843143, -78.9168777, -11.5679207, -65.4155884, 65.2763519
40: -67.7045898, -18.3461914, -67.6355820, -18.3265209, -41.2634888, 41.1334152
41: -55.2039337, -6.8306475, -55.1636086, -6.8209152, -42.3539734, 42.2231216
42: -33.9107971, 6.7825232, -33.9348946, 6.8185549, -37.6679993, 37.6327858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9064911, upper bound: 44.9152451
time: 26.68 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9076066, upper bound: 45.0398400
time: 42.58 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -28.0772896, 17.0264359, -28.1274452, 17.0311623, -44.3252296, 44.3678093
1: -13.6548071, 17.0378990, -13.6837282, 17.0415993, -30.6964073, 30.7216263
2: -14.0663376, 21.6113205, -14.0953026, 21.6150856, -35.4428062, 35.4624405
3: -12.8875313, 23.3928242, -12.9128180, 23.3973179, -36.2848511, 36.3056412
4: -21.5536880, 18.4345112, -21.5803146, 18.4447002, -39.9983902, 40.0148239
5: -11.9872618, 22.7932701, -12.0192280, 22.7981358, -34.7853966, 34.8125000
6: -50.6774750, -3.6130657, -50.6816330, -3.5627894, -40.5664101, 40.5226173
7: -16.3770447, 18.4036636, -16.4088783, 18.4073410, -34.7843857, 34.8125420
8: -18.2845345, 21.2764168, -18.3388710, 21.2823372, -39.5668716, 39.6152878
9: -16.7000713, 23.2265015, -16.7335682, 23.2312431, -38.5802841, 38.6150856
10: -24.2796211, 38.4586334, -24.3169003, 38.4689407, -61.7461853, 61.7894478
11: -24.7608948, 17.5963135, -24.7668152, 17.6102448, -42.3711395, 42.3631287
12: -28.6439629, 20.0923920, -28.6486435, 20.1253796, -46.8963699, 46.8822479
13: -32.9253807, 28.7571411, -32.9439926, 28.7741928, -61.6995735, 61.7011337
14: -23.4603882, 39.1604614, -23.5246239, 39.1649971, -59.9924126, 60.0563698
15: -18.9174614, 25.8389969, -18.9505024, 25.8468704, -44.7643318, 44.7894974
16: -32.7141991, 19.8561192, -32.7428055, 19.8648701, -52.5790710, 52.5989227
17: -17.7489262, 38.4310532, -17.7862320, 38.4357643, -55.2046585, 55.2431870
18: -25.7759380, 19.6146927, -25.7828960, 19.6250954, -45.4010315, 45.3975906
19: -26.3990746, 12.4823694, -26.4077454, 12.5102186, -38.9092941, 38.8901138
20: -21.0738487, 20.4294052, -21.0822620, 20.4554424, -41.5292892, 41.5116653
21: -25.6798515, 18.8695335, -25.6896553, 18.9065571, -44.5864105, 44.5591888
22: -22.0815277, 24.5173950, -22.0892372, 24.5417175, -46.6232452, 46.6066322
23: -21.6813755, 17.4964027, -21.6916313, 17.5055084, -39.1868820, 39.1880341
24: -32.1100311, 11.8953285, -32.1167221, 11.9134569, -44.0234871, 44.0120506
25: -18.0869160, 25.4194221, -18.0984020, 25.4379501, -43.5248642, 43.5178223
26: -29.2147484, 26.9531384, -29.2223034, 26.9758415, -56.1905899, 56.1754417
27: -32.0891800, 16.5391140, -32.0975342, 16.5594864, -47.7858658, 47.7608299
28: -21.5130386, 21.6954918, -21.5216293, 21.7165470, -43.2295837, 43.2171211
29: -23.6813297, 22.2260208, -23.6856422, 22.2389221, -45.9202499, 45.9116631
30: -29.6055660, 16.8453236, -29.6134491, 16.8689461, -45.9325142, 45.9195518
31: -26.3303719, 19.0729237, -26.3435078, 19.1052589, -45.4356308, 45.4164314
32: -42.2125320, 8.4613371, -42.2177429, 8.4971352, -47.5741768, 47.5381279
33: -72.3178940, -5.6489420, -72.3271637, -5.5860920, -61.2993622, 61.2571182
34: -56.4554367, -5.5124340, -56.4610252, -5.4675026, -43.6030693, 43.5639610
35: -50.1051521, 0.0127506, -50.1122665, 0.0675383, -48.2597275, 48.2055206
36: -47.7371941, 4.9159927, -47.7449036, 4.9721317, -52.0150909, 51.9647064
37: -83.6284485, -17.4694500, -83.6373672, -17.4334869, -58.4060364, 58.4205132
38: -58.5971832, 3.1966419, -58.6090736, 3.2640486, -61.2847443, 61.2129440
39: -78.9156342, -11.6268196, -78.9263000, -11.5652237, -65.2910614, 65.2435532
40: -67.6355743, -18.3429985, -67.6431122, -18.3149548, -41.1875114, 41.1964874
41: -55.1680145, -6.8538389, -55.1712990, -6.8159504, -42.3066788, 42.2788506
42: -33.9473648, 6.8110762, -33.9519653, 6.8269758, -37.7140732, 37.6857414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9981410, upper bound: 44.9152451
time: 29.05 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9994222, upper bound: 45.0398400
time: 57.81 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -28.1588097, 17.0799065, -28.1037827, 16.9919205, -44.3728752, 44.3993263
1: -13.6890059, 17.0666580, -13.6650829, 16.9958763, -30.6848831, 30.7317410
2: -14.0987196, 21.6392117, -14.0630274, 21.5581055, -35.4246826, 35.4612465
3: -12.9113789, 23.4136238, -12.8958740, 23.3455582, -36.2569351, 36.3094978
4: -21.5820045, 18.4384880, -21.5438652, 18.3923969, -39.9744034, 39.9823532
5: -12.0204887, 22.8148346, -12.0045147, 22.7560425, -34.7765312, 34.8193512
6: -50.7784729, -3.5592175, -50.6661835, -3.5874424, -40.6273613, 40.5579071
7: -16.4103432, 18.4134521, -16.3800812, 18.3411388, -34.7514801, 34.7935333
8: -18.3384933, 21.3254776, -18.3033409, 21.2219658, -39.5604591, 39.6288185
9: -16.7154465, 23.2791576, -16.6612396, 23.1861877, -38.5501328, 38.6093712
10: -24.3112259, 38.5321846, -24.2588024, 38.4112587, -61.7226868, 61.8190308
11: -24.7863922, 17.6010246, -24.7459717, 17.5861053, -42.3724976, 42.3469963
12: -28.6226540, 20.1335030, -28.5323410, 20.0585938, -46.8047256, 46.8180580
13: -32.9240036, 28.7974873, -32.8677139, 28.7203674, -61.6443710, 61.6651993
14: -23.5095005, 39.2364235, -23.3766232, 39.1181717, -59.9927063, 60.0142593
15: -18.9610519, 25.8822861, -18.9201088, 25.8321609, -44.7932129, 44.8023949
16: -32.7429008, 19.9095840, -32.7108879, 19.8315201, -52.5744209, 52.6204720
17: -17.7743359, 38.5017014, -17.6965847, 38.4104233, -55.2039680, 55.2457924
18: -25.8364906, 19.6219978, -25.7331085, 19.6086807, -45.4451714, 45.3551064
19: -26.4804420, 12.5123463, -26.3817234, 12.5080595, -38.9885025, 38.8940697
20: -21.1479416, 20.4607506, -21.0317478, 20.4368000, -41.5847397, 41.4925003
21: -25.7712879, 18.9096146, -25.6415329, 18.8904934, -44.6617813, 44.5511475
22: -22.1506138, 24.5415573, -22.0429993, 24.5250969, -46.6757126, 46.5845566
23: -21.7106304, 17.5133553, -21.6636200, 17.4974213, -39.2080536, 39.1769753
24: -32.2043839, 11.8997841, -32.0752106, 11.8753414, -44.0797272, 43.9749947
25: -18.1338673, 25.4309349, -18.0755730, 25.4236488, -43.5575180, 43.5065079
26: -29.2417068, 26.9702320, -29.1424980, 26.9368896, -56.1785965, 56.1127319
27: -32.1454048, 16.5489159, -32.0499229, 16.5332298, -47.8667374, 47.7254105
28: -21.5522747, 21.7146626, -21.4909668, 21.7076683, -43.2599411, 43.2056274
29: -23.7313499, 22.2351723, -23.6354904, 22.2118416, -45.9431915, 45.8706627
30: -29.6871986, 16.8734207, -29.5984497, 16.8471279, -45.9828873, 45.9318581
31: -26.4409351, 19.1042786, -26.3033676, 19.0840111, -45.5249481, 45.4076462
32: -42.2737656, 8.5048790, -42.1774864, 8.4647255, -47.5748062, 47.5401688
33: -72.4641571, -5.6060562, -72.2866440, -5.6568022, -61.3863220, 61.2576828
34: -56.5460434, -5.4695988, -56.4269104, -5.4899406, -43.6746140, 43.5648651
35: -50.2211113, 0.0590925, -50.0817795, 0.0451574, -48.3539276, 48.1934738
36: -47.8176765, 4.9723606, -47.6856651, 4.9547758, -52.0687180, 51.9562531
37: -83.7331161, -17.4413280, -83.6005707, -17.4703808, -58.4783325, 58.4167252
38: -58.7457352, 3.2640324, -58.5739632, 3.2555037, -61.3981628, 61.2364273
39: -79.0746078, -11.5699463, -78.8959122, -11.5865822, -65.4398499, 65.2713242
40: -67.7241898, -18.3313217, -67.6114197, -18.3755264, -41.2259483, 41.2122993
41: -55.2243462, -6.8122549, -55.1443977, -6.8395815, -42.2988358, 42.3181000
42: -33.9580460, 6.8264875, -33.9516602, 6.7946720, -37.6646423, 37.6997108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0361989, upper bound: 44.7624539
time: 64.83 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0374999, upper bound: 44.8875702
time: 29.57 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -28.1641998, 17.0923901, -28.1312218, 17.0310650, -44.4056244, 44.4397964
1: -13.6912718, 17.0821896, -13.6860247, 17.0414734, -30.7327461, 30.7682152
2: -14.1018105, 21.6585922, -14.0970764, 21.6150475, -35.4753036, 35.5155106
3: -12.9140320, 23.4314480, -12.9135437, 23.3972778, -36.3113098, 36.3449936
4: -21.5863800, 18.4557190, -21.5810623, 18.4438438, -40.0302238, 40.0367813
5: -12.0224762, 22.8291721, -12.0207787, 22.7981453, -34.8206215, 34.8499527
6: -50.7847824, -3.5538349, -50.6814194, -3.5612464, -40.6847649, 40.5781097
7: -16.4137154, 18.4374695, -16.4104881, 18.4073410, -34.8210564, 34.8479576
8: -18.3432732, 21.3461781, -18.3417110, 21.2824936, -39.6257668, 39.6878891
9: -16.7394905, 23.2840157, -16.7337685, 23.2310925, -38.6172523, 38.6709747
10: -24.3305893, 38.5396194, -24.3179913, 38.4691200, -61.7989349, 61.8784561
11: -24.7952499, 17.6050453, -24.7665672, 17.6076965, -42.4029465, 42.3716125
12: -28.6625919, 20.1393776, -28.6484871, 20.1269226, -46.9134369, 46.9267426
13: -32.9488983, 28.8050804, -32.9431877, 28.7746506, -61.7235489, 61.7482681
14: -23.5590611, 39.2391510, -23.5267086, 39.1649551, -60.0900841, 60.1384735
15: -18.9680901, 25.8922157, -18.9513779, 25.8467216, -44.8148117, 44.8435936
16: -32.7523537, 19.9165497, -32.7426605, 19.8651314, -52.6174850, 52.6592102
17: -17.8047009, 38.5051842, -17.7877083, 38.4355774, -55.2596817, 55.3229790
18: -25.8461475, 19.6263123, -25.7831516, 19.6232681, -45.4694138, 45.4094620
19: -26.4880848, 12.5135460, -26.4077187, 12.5121689, -39.0002518, 38.9212646
20: -21.1639576, 20.4626350, -21.0823517, 20.4574814, -41.6214371, 41.5449867
21: -25.7871265, 18.9118614, -25.6897926, 18.9091663, -44.6962929, 44.6016541
22: -22.1653805, 24.5443993, -22.0894508, 24.5435715, -46.7089539, 46.6338501
23: -21.7176418, 17.5157852, -21.6911240, 17.5058765, -39.2235184, 39.2069092
24: -32.2095604, 11.9117985, -32.1169281, 11.9119787, -44.1215401, 44.0287247
25: -18.1408157, 25.4337502, -18.0986481, 25.4352036, -43.5760193, 43.5323982
26: -29.2669010, 26.9727268, -29.2224865, 26.9729538, -56.2398529, 56.1952133
27: -32.1531410, 16.5575809, -32.0977097, 16.5585136, -47.8757248, 47.7828903
28: -21.5608673, 21.7161942, -21.5216713, 21.7141342, -43.2750015, 43.2378654
29: -23.7492695, 22.2372971, -23.6856537, 22.2379971, -45.9872665, 45.9229507
30: -29.6920509, 16.8772697, -29.6134319, 16.8679085, -46.0133667, 45.9511528
31: -26.4495316, 19.1118965, -26.3435326, 19.1079350, -45.5574646, 45.4554291
32: -42.2872429, 8.5092421, -42.2176132, 8.4992304, -47.6510391, 47.5843353
33: -72.4680786, -5.5820646, -72.3273621, -5.5825958, -61.4532166, 61.3238525
34: -56.5517464, -5.4614344, -56.4610443, -5.4649086, -43.7072716, 43.6134491
35: -50.2269287, 0.0673180, -50.1123123, 0.0707359, -48.3853912, 48.2604141
36: -47.8378792, 4.9750404, -47.7450485, 4.9753180, -52.1209259, 52.0234909
37: -83.7414703, -17.4285183, -83.6370926, -17.4316711, -58.5233231, 58.4605293
38: -58.7558784, 3.2668476, -58.6094437, 3.2675657, -61.4468384, 61.2835693
39: -79.0799408, -11.5618649, -78.9260559, -11.5615807, -65.4606018, 65.3083191
40: -67.7294388, -18.3099098, -67.6430893, -18.3136292, -41.2844543, 41.2278709
41: -55.2297668, -6.8045053, -55.1709785, -6.8138475, -42.3747444, 42.3275299
42: -33.9610596, 6.8311234, -33.9516907, 6.8252277, -37.7282333, 37.7073593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0385445, upper bound: 44.9152451
time: 25.87 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0398400, upper bound: 45.0398400
time: 48.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 76.60 seconds
IS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 76.60
Output dim: 14, lower bound: -44.8642075, upper bound: 44.9152451
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -44.8671733, upper bound: 45.0398400
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 76.60
Output dim: 14, lower bound: -44.9064911, upper bound: 44.9152451
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -44.9076066, upper bound: 45.0398400
IS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 76.60
Output dim: 14, lower bound: -44.9981410, upper bound: 44.9152451
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -44.9994222, upper bound: 45.0398400
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -45.0361989, upper bound: 44.7624539
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -45.0374999, upper bound: 44.8875702
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -45.0385445, upper bound: 44.9152451
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 76.60
Output dim: 14, lower bound: -45.0398400, upper bound: 45.0398400

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -28.0285797, 16.9680214, -28.1202927, 17.0112057, -44.2508163, 44.3134537
1: -13.6166620, 16.9581299, -13.6806412, 17.0135231, -30.6301842, 30.6387711
2: -14.0097132, 21.5129776, -14.0907078, 21.5804005, -35.3431015, 35.3746872
3: -12.8444271, 23.3110619, -12.9093838, 23.3671761, -36.2116013, 36.2204437
4: -21.4883938, 18.3246498, -21.5726662, 18.4062347, -39.8946304, 39.8973160
5: -11.9427776, 22.7203884, -12.0159121, 22.7692127, -34.7119904, 34.7363014
6: -50.6428375, -3.6477394, -50.6698990, -3.5776625, -40.5500031, 40.4430313
7: -16.3267746, 18.3134003, -16.4045506, 18.3747482, -34.7015228, 34.7179489
8: -18.2238159, 21.1813984, -18.3315716, 21.2479973, -39.4718132, 39.5129700
9: -16.5953865, 23.1627178, -16.6982937, 23.2225189, -38.4858475, 38.4993782
10: -24.1343346, 38.3636360, -24.2672596, 38.4568138, -61.6109467, 61.6264610
11: -24.6917858, 17.5550575, -24.7421036, 17.6039276, -42.2957153, 42.2971611
12: -28.4857864, 19.9983845, -28.5928040, 20.1166210, -46.7505798, 46.7292976
13: -32.8683167, 28.7139225, -32.9224396, 28.7635612, -61.6318779, 61.6363602
14: -23.2928734, 39.1016464, -23.4672966, 39.1601486, -59.8515244, 59.9128151
15: -18.8635712, 25.7894535, -18.9393177, 25.8291187, -44.6926880, 44.7287712
16: -32.6540909, 19.8003273, -32.7228699, 19.8545475, -52.5086365, 52.5231972
17: -17.6434631, 38.3901978, -17.7493553, 38.4297066, -55.1110535, 55.1511612
18: -25.7031155, 19.5640507, -25.7678242, 19.6098289, -45.3129425, 45.3318748
19: -26.3496933, 12.4646301, -26.3896999, 12.5082550, -38.8579483, 38.8543320
20: -21.0053654, 20.3987999, -21.0595627, 20.4527664, -41.4581299, 41.4583626
21: -25.6026936, 18.8342381, -25.6625080, 18.9032097, -44.5059052, 44.4967461
22: -22.0272408, 24.4937897, -22.0688210, 24.5369034, -46.5641441, 46.5626106
23: -21.6434574, 17.4767303, -21.6780777, 17.5029945, -39.1464539, 39.1548080
24: -32.0665588, 11.8483219, -32.1084671, 11.8981628, -43.9647217, 43.9567871
25: -18.0396595, 25.3922253, -18.0809784, 25.4341011, -43.4737625, 43.4732056
26: -29.1275425, 26.8943882, -29.1894684, 26.9713745, -56.0989151, 56.0838547
27: -32.0305023, 16.4940548, -32.0862656, 16.5433159, -47.6903000, 47.7366638
28: -21.4629955, 21.6718578, -21.5044708, 21.7135582, -43.1765518, 43.1763306
29: -23.6086922, 22.1859074, -23.6584969, 22.2349510, -45.8436432, 45.8444061
30: -29.5461826, 16.8054352, -29.5922775, 16.8629322, -45.8691788, 45.8484726
31: -26.2784100, 19.0562534, -26.3268776, 19.1015625, -45.3799744, 45.3831329
32: -42.1396980, 8.4188242, -42.1928177, 8.4875078, -47.5011024, 47.4482460
33: -72.2791748, -5.7217255, -72.3212509, -5.6097727, -61.2253723, 61.1749268
34: -56.4262085, -5.5406961, -56.4544258, -5.4767771, -43.5429764, 43.5271606
35: -50.0768166, -0.0186567, -50.1054306, 0.0566435, -48.1761932, 48.1769028
36: -47.6873512, 4.9038486, -47.7272301, 4.9659548, -51.9493866, 51.9252701
37: -83.5894699, -17.4989548, -83.6251450, -17.4445229, -58.3499451, 58.2988968
38: -58.5386314, 3.1730499, -58.5948067, 3.2550707, -61.2068863, 61.1748123
39: -78.8755798, -11.6494703, -78.9157791, -11.5724201, -65.2442932, 65.2106094
40: -67.6105347, -18.3794556, -67.6349030, -18.3285656, -41.1682816, 41.0955582
41: -55.1420708, -6.8805981, -55.1634026, -6.8258305, -42.2833023, 42.1695633
42: -33.8970451, 6.7616186, -33.9346733, 6.8162518, -37.6712189, 37.6072083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8671733, upper bound: 45.0201639
time: 34.83 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8671733, upper bound: 45.0398400
time: 41.53 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -28.1154976, 17.0339737, -28.1240540, 17.0111122, -44.3312225, 44.3854332
1: -13.6531677, 17.0024014, -13.6829281, 17.0133858, -30.6665535, 30.6853294
2: -14.0452061, 21.5602264, -14.0924797, 21.5803375, -35.3755913, 35.4277496
3: -12.8709326, 23.3496895, -12.9100981, 23.3671303, -36.2380638, 36.2597885
4: -21.5211182, 18.3458538, -21.5734043, 18.4053936, -39.9265137, 39.9192581
5: -11.9780331, 22.7563133, -12.0174761, 22.7692318, -34.7472649, 34.7737885
6: -50.7501221, -3.5885024, -50.6696625, -3.5761037, -40.6683960, 40.4985123
7: -16.3634548, 18.3471909, -16.4061813, 18.3747406, -34.7381973, 34.7533722
8: -18.2825451, 21.2511253, -18.3343964, 21.2481480, -39.5306931, 39.5855217
9: -16.6348019, 23.2202511, -16.6985035, 23.2223625, -38.5228195, 38.5552673
10: -24.1852970, 38.4446259, -24.2683144, 38.4569702, -61.6636963, 61.7155037
11: -24.7261219, 17.5637989, -24.7418385, 17.6014175, -42.3275375, 42.3056374
12: -28.5044594, 20.0453243, -28.5926552, 20.1181717, -46.7676697, 46.7737808
13: -32.8918114, 28.7618217, -32.9216385, 28.7639961, -61.6558075, 61.6834602
14: -23.3916130, 39.1803513, -23.4694138, 39.1600952, -59.9492378, 59.9949875
15: -18.9141579, 25.8426800, -18.9402046, 25.8289795, -44.7431374, 44.7828827
16: -32.6922302, 19.8607597, -32.7227173, 19.8547935, -52.5470238, 52.5834770
17: -17.6992741, 38.4643478, -17.7508316, 38.4294891, -55.1660957, 55.2309570
18: -25.7733917, 19.5756893, -25.7680836, 19.6080017, -45.3813934, 45.3437729
19: -26.4387016, 12.4957981, -26.3896694, 12.5101681, -38.9488678, 38.8854675
20: -21.0954723, 20.4320374, -21.0596428, 20.4547920, -41.5502625, 41.4916801
21: -25.7099590, 18.8765163, -25.6626205, 18.9058132, -44.6157722, 44.5391388
22: -22.1111126, 24.5207748, -22.0690365, 24.5387287, -46.6498413, 46.5898132
23: -21.6797028, 17.4960957, -21.6775723, 17.5033855, -39.1830902, 39.1736679
24: -32.1661148, 11.8647709, -32.1086807, 11.8967018, -44.0628166, 43.9734497
25: -18.0935707, 25.4065781, -18.0812206, 25.4313660, -43.5249367, 43.4878006
26: -29.1795883, 26.9139748, -29.1896763, 26.9684658, -56.1480560, 56.1036530
27: -32.0945740, 16.5124836, -32.0864563, 16.5423450, -47.7802048, 47.7587433
28: -21.5108032, 21.6925583, -21.5045147, 21.7111511, -43.2219543, 43.1970749
29: -23.6766090, 22.1972046, -23.6585102, 22.2340488, -45.9106598, 45.8557129
30: -29.6326313, 16.8374062, -29.5922241, 16.8618984, -45.9500198, 45.8800812
31: -26.3975849, 19.0952339, -26.3268986, 19.1042213, -45.5018082, 45.4221344
32: -42.2143898, 8.4667292, -42.1927147, 8.4895887, -47.5779572, 47.4944115
33: -72.4293594, -5.6548233, -72.3214111, -5.6062508, -61.3792801, 61.2416382
34: -56.5225182, -5.4896536, -56.4544563, -5.4742002, -43.6471558, 43.5766716
35: -50.1986122, 0.0359278, -50.1054993, 0.0598602, -48.3019104, 48.2318077
36: -47.7880249, 4.9628544, -47.7273598, 4.9691448, -52.0552979, 51.9840469
37: -83.7025146, -17.4580078, -83.6249390, -17.4427032, -58.4671936, 58.3389206
38: -58.6973343, 3.2432957, -58.5951385, 3.2585707, -61.3690186, 61.2454910
39: -79.0399246, -11.5844860, -78.9155655, -11.5687551, -65.4138031, 65.2754059
40: -67.7044373, -18.3463478, -67.6348877, -18.3272285, -41.2651711, 41.1269646
41: -55.2038231, -6.8312473, -55.1630974, -6.8237143, -42.3513680, 42.2182732
42: -33.9106865, 6.7816324, -33.9343834, 6.8145142, -37.6853676, 37.6288109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7533869, upper bound: 44.9830142
time: 24.86 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9055786, upper bound: 45.0392273
time: 43.68 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -28.0770760, 17.0262833, -28.1264076, 17.0304680, -44.3193932, 44.3665314
1: -13.6547165, 17.0377121, -13.6833286, 17.0407734, -30.6954899, 30.7210407
2: -14.0662527, 21.6111450, -14.0949545, 21.6142769, -35.4340973, 35.4618568
3: -12.8874397, 23.3922691, -12.9124222, 23.3947525, -36.2821922, 36.3046913
4: -21.5535297, 18.4342957, -21.5796432, 18.4436913, -39.9972229, 40.0139389
5: -11.9871778, 22.7924099, -12.0188789, 22.7941685, -34.7813454, 34.8112869
6: -50.6773453, -3.6148510, -50.6810379, -3.5709977, -40.5932693, 40.5176239
7: -16.3769054, 18.4033203, -16.4083290, 18.4057617, -34.7826691, 34.8116493
8: -18.2843628, 21.2760792, -18.3381310, 21.2807369, -39.5651016, 39.6142120
9: -16.6996784, 23.2263870, -16.7317352, 23.2307262, -38.5793304, 38.5979195
10: -24.2791519, 38.4584503, -24.3147202, 38.4680252, -61.7447357, 61.7700768
11: -24.7605534, 17.5961418, -24.7652683, 17.6094475, -42.3700027, 42.3614120
12: -28.6437073, 20.0922279, -28.6474476, 20.1247139, -46.8954048, 46.8775253
13: -32.9246826, 28.7569008, -32.9411011, 28.7729797, -61.6976624, 61.6980019
14: -23.4595776, 39.1603928, -23.5209942, 39.1646500, -59.9911880, 60.0271530
15: -18.9171906, 25.8386803, -18.9493294, 25.8454323, -44.7626228, 44.7880096
16: -32.7138214, 19.8559761, -32.7410431, 19.8642178, -52.5780411, 52.5970192
17: -17.7485886, 38.4309616, -17.7847214, 38.4353561, -55.2028809, 55.2275352
18: -25.7757397, 19.6145744, -25.7819424, 19.6246147, -45.4003525, 45.3965149
19: -26.3984699, 12.4823284, -26.4049778, 12.5100698, -38.9085388, 38.8873062
20: -21.0735779, 20.4293156, -21.0810413, 20.4550591, -41.5286369, 41.5103569
21: -25.6795368, 18.8694763, -25.6882286, 18.9062824, -44.5858192, 44.5577049
22: -22.0811653, 24.5172749, -22.0876160, 24.5411263, -46.6222916, 46.6048889
23: -21.6809864, 17.4963188, -21.6898136, 17.5051498, -39.1861343, 39.1861343
24: -32.1098175, 11.8952475, -32.1157646, 11.9130640, -44.0228806, 44.0110130
25: -18.0865173, 25.4193325, -18.0966644, 25.4375973, -43.5241165, 43.5159988
26: -29.2137260, 26.9530411, -29.2175617, 26.9753990, -56.1891251, 56.1706009
27: -32.0889740, 16.5388374, -32.0966530, 16.5581665, -47.7616730, 47.7596436
28: -21.5126724, 21.6953621, -21.5198784, 21.7160759, -43.2287483, 43.2152405
29: -23.6808815, 22.2259083, -23.6836128, 22.2384777, -45.9193573, 45.9095230
30: -29.6053009, 16.8451939, -29.6122246, 16.8683529, -45.9314117, 45.9100990
31: -26.3300781, 19.0728741, -26.3422203, 19.1050625, -45.4351425, 45.4150925
32: -42.2123718, 8.4603176, -42.2170181, 8.4924765, -47.5781631, 47.5330582
33: -72.3175964, -5.6493187, -72.3258972, -5.5878057, -61.2892761, 61.2553177
34: -56.4553223, -5.5128241, -56.4604111, -5.4692898, -43.5923233, 43.5630608
35: -50.1049728, 0.0124388, -50.1113892, 0.0661955, -48.2566833, 48.2195358
36: -47.7370644, 4.9153728, -47.7443123, 4.9692850, -52.0075531, 51.9634247
37: -83.6281967, -17.4695854, -83.6361847, -17.4341354, -58.4057770, 58.4144058
38: -58.5970421, 3.1962442, -58.6083984, 3.2622414, -61.2660217, 61.2118225
39: -78.9153214, -11.6270199, -78.9250107, -11.5660458, -65.2893677, 65.2425919
40: -67.6354218, -18.3431606, -67.6424103, -18.3156548, -41.1892242, 41.1900330
41: -55.1679039, -6.8544378, -55.1707954, -6.8187523, -42.3040466, 42.2740135
42: -33.9472618, 6.8102007, -33.9514618, 6.8229427, -37.7314682, 37.6817856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8642075, upper bound: 45.0201639
time: 47.95 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8642075, upper bound: 45.0398400
time: 48.09 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -28.1551628, 17.0743580, -28.0841904, 16.9751835, -44.3497810, 44.3729248
1: -13.6874599, 17.0591049, -13.6520977, 16.9734993, -30.6609592, 30.7112026
2: -14.0967646, 21.6277294, -14.0402937, 21.5246201, -35.3890114, 35.4283905
3: -12.9099016, 23.3960190, -12.8701000, 23.2953606, -36.2052612, 36.2661209
4: -21.5785713, 18.4252510, -21.5180168, 18.3540535, -39.9326248, 39.9432678
5: -12.0187483, 22.8014164, -11.9822578, 22.7177734, -34.7365227, 34.7836761
6: -50.7753677, -3.5711970, -50.6542397, -3.6234546, -40.6007805, 40.5585747
7: -16.4075851, 18.4005394, -16.3544540, 18.3040867, -34.7116699, 34.7549934
8: -18.3335419, 21.3132248, -18.2748909, 21.1858482, -39.5193901, 39.5881157
9: -16.6949730, 23.2775612, -16.6000538, 23.1553879, -38.5002518, 38.5505905
10: -24.2748699, 38.5295143, -24.1516724, 38.3546524, -61.6285553, 61.7068176
11: -24.7705975, 17.5988941, -24.7004700, 17.5643463, -42.3349457, 42.2993622
12: -28.6055431, 20.1304150, -28.4828110, 20.0192337, -46.7485046, 46.7659798
13: -32.9144363, 28.7922020, -32.8368721, 28.7008400, -61.6152763, 61.6290741
14: -23.4723625, 39.2351837, -23.2662716, 39.0822067, -59.9186440, 59.9015923
15: -18.9560909, 25.8738213, -18.8989124, 25.8067303, -44.7628212, 44.7727356
16: -32.7220306, 19.9064636, -32.6467285, 19.7934456, -52.5154762, 52.5531921
17: -17.7637501, 38.4994087, -17.6647568, 38.3932915, -55.1742287, 55.2105103
18: -25.8311405, 19.6190567, -25.7051105, 19.5991344, -45.4302750, 45.3241653
19: -26.4714165, 12.5116606, -26.3538494, 12.4985847, -38.9700012, 38.8655090
20: -21.1409988, 20.4597664, -21.0093994, 20.4278297, -41.5688286, 41.4691658
21: -25.7581825, 18.9087982, -25.6023598, 18.8783188, -44.6365013, 44.5111580
22: -22.1453247, 24.5378838, -22.0244598, 24.5122128, -46.6575394, 46.5623436
23: -21.7006226, 17.5119915, -21.6341114, 17.4843273, -39.1849518, 39.1461029
24: -32.2005005, 11.8983727, -32.0614777, 11.8700285, -44.0705299, 43.9598503
25: -18.1216507, 25.4296741, -18.0391541, 25.4078484, -43.5294991, 43.4688263
26: -29.2323494, 26.9671745, -29.1117096, 26.9123764, -56.1447258, 56.0788841
27: -32.1414185, 16.5364265, -32.0194817, 16.4969501, -47.8237457, 47.6817055
28: -21.5457726, 21.7117062, -21.4705639, 21.6980839, -43.2438583, 43.1822701
29: -23.7263355, 22.2325172, -23.6189575, 22.2019405, -45.9282761, 45.8514748
30: -29.6764526, 16.8712082, -29.5660133, 16.8281555, -45.9507370, 45.8908539
31: -26.4321671, 19.1034355, -26.2739811, 19.0779743, -45.5101395, 45.3774185
32: -42.2700233, 8.4978952, -42.1665077, 8.4441395, -47.5491409, 47.5232010
33: -72.4611816, -5.6279659, -72.2567978, -5.7209148, -61.3190765, 61.2053604
34: -56.5432663, -5.4870739, -56.4010391, -5.5431528, -43.6179886, 43.5211601
35: -50.2185593, 0.0428381, -50.0594978, -0.0025959, -48.3031540, 48.1442108
36: -47.8142624, 4.9570704, -47.6658478, 4.9110308, -52.0204620, 51.9212189
37: -83.7292480, -17.4459362, -83.5847397, -17.4846363, -58.4523010, 58.3907089
38: -58.7417068, 3.2443762, -58.5357361, 3.1977339, -61.3326416, 61.1768265
39: -79.0686951, -11.5785980, -78.8668976, -11.6120167, -65.4096298, 65.2331238
40: -67.7214355, -18.3408051, -67.5901947, -18.4032516, -41.1980743, 41.1913872
41: -55.2220497, -6.8255234, -55.1235275, -6.8795671, -42.2619286, 42.2968750
42: -33.9555359, 6.8214054, -33.9471741, 6.7762585, -37.6348343, 37.6937180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9764213, upper bound: 44.5880704
time: 52.50 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0357106, upper bound: 44.7619647
time: 24.61 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -28.1585693, 17.0797329, -28.1027222, 16.9912338, -44.3670235, 44.3980408
1: -13.6889095, 17.0664787, -13.6646976, 16.9950676, -30.6839771, 30.7311764
2: -14.0986347, 21.6390343, -14.0626831, 21.5573044, -35.4159508, 35.4606476
3: -12.9112873, 23.4130630, -12.8954687, 23.3429794, -36.2542648, 36.3085327
4: -21.5818691, 18.4382706, -21.5431671, 18.3913937, -39.9732628, 39.9814377
5: -12.0204105, 22.8139744, -12.0041733, 22.7520847, -34.7724953, 34.8181458
6: -50.7783394, -3.5609956, -50.6655426, -3.5956497, -40.6542244, 40.5529060
7: -16.4102173, 18.4131126, -16.3795319, 18.3395729, -34.7497902, 34.7926445
8: -18.3383160, 21.3251266, -18.3025742, 21.2203674, -39.5586853, 39.6277008
9: -16.7150383, 23.2790394, -16.6593895, 23.1856613, -38.5491562, 38.5922089
10: -24.3107452, 38.5319862, -24.2566109, 38.4103546, -61.7212067, 61.7996140
11: -24.7860432, 17.6008472, -24.7444305, 17.5853271, -42.3713684, 42.3452759
12: -28.6223869, 20.1333447, -28.5311623, 20.0579243, -46.8037834, 46.8133698
13: -32.9232864, 28.7972031, -32.8648262, 28.7191410, -61.6424255, 61.6620293
14: -23.5086899, 39.2363510, -23.3729649, 39.1178131, -59.9914780, 59.9850807
15: -18.9607868, 25.8819695, -18.9189224, 25.8307228, -44.7915115, 44.8008919
16: -32.7425194, 19.9094429, -32.7091217, 19.8308735, -52.5733948, 52.6185646
17: -17.7739944, 38.5016098, -17.6950951, 38.4100266, -55.2021828, 55.2301636
18: -25.8362656, 19.6218872, -25.7321663, 19.6081924, -45.4444580, 45.3540535
19: -26.4798298, 12.5123119, -26.3789482, 12.5079107, -38.9877396, 38.8912582
20: -21.1476669, 20.4606819, -21.0305290, 20.4364281, -41.5840950, 41.4912109
21: -25.7709713, 18.9095402, -25.6400948, 18.8902283, -44.6612015, 44.5496368
22: -22.1502724, 24.5414162, -22.0413818, 24.5244789, -46.6747513, 46.5827980
23: -21.7102146, 17.5132713, -21.6618004, 17.4970398, -39.2072525, 39.1750717
24: -32.2041702, 11.8996868, -32.0742683, 11.8749466, -44.0791168, 43.9739532
25: -18.1334877, 25.4308662, -18.0738239, 25.4232750, -43.5567627, 43.5046921
26: -29.2406807, 26.9701233, -29.1377907, 26.9364243, -56.1771049, 56.1079140
27: -32.1452026, 16.5486279, -32.0490417, 16.5319099, -47.8425674, 47.7242203
28: -21.5518875, 21.7145672, -21.4892120, 21.7071972, -43.2590866, 43.2037811
29: -23.7309151, 22.2350655, -23.6334553, 22.2114048, -45.9423218, 45.8685226
30: -29.6869240, 16.8732948, -29.5972099, 16.8464890, -45.9817924, 45.9223976
31: -26.4406376, 19.1042366, -26.3020725, 19.0838051, -45.5244446, 45.4063110
32: -42.2736053, 8.5038567, -42.1767578, 8.4600563, -47.5787582, 47.5350723
33: -72.4638596, -5.6064339, -72.2853928, -5.6584702, -61.3762131, 61.2558975
34: -56.5458946, -5.4699869, -56.4262695, -5.4917459, -43.6638870, 43.5639496
35: -50.2209053, 0.0588036, -50.0809212, 0.0437994, -48.3508606, 48.2074738
36: -47.8175507, 4.9717474, -47.6850967, 4.9519386, -52.0611420, 51.9549637
37: -83.7328568, -17.4414902, -83.5994339, -17.4710350, -58.4780807, 58.4106293
38: -58.7455750, 3.2636280, -58.5732803, 3.2536955, -61.3795090, 61.2352982
39: -79.0743179, -11.5701447, -78.8946075, -11.5873842, -65.4380798, 65.2703629
40: -67.7240372, -18.3314877, -67.6107330, -18.3762245, -41.2276688, 41.2058411
41: -55.2242432, -6.8128643, -55.1438980, -6.8423338, -42.2962341, 42.3132553
42: -33.9579315, 6.8256016, -33.9511642, 6.7906256, -37.6820297, 37.6957512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9780646, upper bound: 44.7130292
time: 53.42 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0370111, upper bound: 44.8870783
time: 50.52 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -28.1605606, 17.0868320, -28.1116257, 17.0142975, -44.3825188, 44.4134064
1: -13.6897335, 17.0746307, -13.6730309, 17.0190697, -30.7088032, 30.7476616
2: -14.0998583, 21.6471138, -14.0743284, 21.5815506, -35.4395981, 35.4826622
3: -12.9125462, 23.4138508, -12.8877535, 23.3470688, -36.2596130, 36.3016052
4: -21.5829315, 18.4424953, -21.5552406, 18.4054832, -39.9884148, 39.9977341
5: -12.0207367, 22.8157616, -11.9985332, 22.7598133, -34.7805481, 34.8142929
6: -50.7816544, -3.5658298, -50.6695862, -3.5972371, -40.6581955, 40.5787888
7: -16.4109402, 18.4245491, -16.3848686, 18.3702774, -34.7812195, 34.8094177
8: -18.3383236, 21.3339195, -18.3132763, 21.2463837, -39.5847092, 39.6471939
9: -16.7190323, 23.2824173, -16.6725807, 23.2002869, -38.5674095, 38.6121902
10: -24.2942257, 38.5369568, -24.2108078, 38.4125443, -61.7048264, 61.7663002
11: -24.7794609, 17.6029263, -24.7210426, 17.5859375, -42.3653984, 42.3239670
12: -28.6454792, 20.1363029, -28.5989075, 20.0875969, -46.8572311, 46.8746262
13: -32.9393539, 28.7998123, -32.9123459, 28.7551727, -61.6945267, 61.7121582
14: -23.5219402, 39.2379456, -23.4163284, 39.1289902, -60.0160027, 60.0257454
15: -18.9631577, 25.8837509, -18.9302158, 25.8213100, -44.7844696, 44.8139648
16: -32.7314758, 19.9134617, -32.6784821, 19.8270435, -52.5585175, 52.5919418
17: -17.7941246, 38.5029030, -17.7558479, 38.4184303, -55.2299576, 55.2876778
18: -25.8408241, 19.6233788, -25.7552052, 19.6137085, -45.4545326, 45.3785858
19: -26.4790611, 12.5128708, -26.3798428, 12.5026789, -38.9817390, 38.8927155
20: -21.1570339, 20.4616451, -21.0599804, 20.4484806, -41.6055145, 41.5216255
21: -25.7740116, 18.9110336, -25.6505909, 18.8969955, -44.6710052, 44.5616226
22: -22.1600780, 24.5407028, -22.0708313, 24.5307045, -46.6907806, 46.6115341
23: -21.7076416, 17.5144005, -21.6615753, 17.4928131, -39.2004547, 39.1759758
24: -32.2056808, 11.9103928, -32.1032333, 11.9066725, -44.1123543, 44.0136261
25: -18.1286011, 25.4324741, -18.0622177, 25.4194145, -43.5480156, 43.4946899
26: -29.2575474, 26.9696579, -29.1916161, 26.9484634, -56.2060089, 56.1612740
27: -32.1491661, 16.5450974, -32.0673027, 16.5222454, -47.8327560, 47.7392349
28: -21.5543652, 21.7132378, -21.5012493, 21.7045441, -43.2589111, 43.2144852
29: -23.7442341, 22.2346611, -23.6690788, 22.2281189, -45.9723511, 45.9037399
30: -29.6813011, 16.8750610, -29.5809746, 16.8489571, -45.9812393, 45.9101410
31: -26.4407578, 19.1110420, -26.3141613, 19.1018944, -45.5426521, 45.4252014
32: -42.2835159, 8.5022917, -42.2066536, 8.4786205, -47.6253967, 47.5673561
33: -72.4651184, -5.6039762, -72.2975082, -5.6467514, -61.3859711, 61.2715073
34: -56.5489731, -5.4789057, -56.4352303, -5.5180845, -43.6506195, 43.5697594
35: -50.2243919, 0.0510740, -50.0900574, 0.0230074, -48.3346481, 48.2111549
36: -47.8344574, 4.9597282, -47.7252159, 4.9315233, -52.0727539, 51.9884109
37: -83.7376099, -17.4331131, -83.6213226, -17.4459457, -58.4972992, 58.4345055
38: -58.7518387, 3.2471390, -58.5712395, 3.2097998, -61.3813171, 61.2240372
39: -79.0740280, -11.5705214, -78.8970795, -11.5870695, -65.4303818, 65.2701874
40: -67.7266388, -18.3193684, -67.6218567, -18.3413658, -41.2565613, 41.2069740
41: -55.2274704, -6.8178186, -55.1501007, -6.8538914, -42.3377800, 42.3063316
42: -33.9585419, 6.8260241, -33.9471664, 6.8068151, -37.6984329, 37.7013550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9817017, upper bound: 44.7460528
time: 24.69 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0379675, upper bound: 44.9146438
time: 50.54 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -28.1639709, 17.0922165, -28.1301575, 17.0303764, -44.3997612, 44.4385109
1: -13.6911879, 17.0820007, -13.6856298, 17.0406456, -30.7318344, 30.7676315
2: -14.1017084, 21.6584167, -14.0966959, 21.6142349, -35.4665756, 35.5149155
3: -12.9139395, 23.4308968, -12.9131413, 23.3947048, -36.3086433, 36.3440399
4: -21.5862255, 18.4555130, -21.5803566, 18.4428520, -40.0290756, 40.0358696
5: -12.0224047, 22.8283195, -12.0204306, 22.7941933, -34.8165970, 34.8487511
6: -50.7846298, -3.5556145, -50.6808014, -3.5694480, -40.7116394, 40.5731430
7: -16.4135857, 18.4371357, -16.4099350, 18.4057713, -34.8193588, 34.8470688
8: -18.3431015, 21.3458290, -18.3409538, 21.2808914, -39.6239929, 39.6867828
9: -16.7390900, 23.2838955, -16.7319298, 23.2305584, -38.6163101, 38.6537971
10: -24.3301029, 38.5394440, -24.3157692, 38.4682083, -61.7974548, 61.8590889
11: -24.7949104, 17.6048737, -24.7649994, 17.6069107, -42.4018211, 42.3698730
12: -28.6623459, 20.1392345, -28.6473045, 20.1262474, -46.9124756, 46.9220276
13: -32.9481964, 28.8048096, -32.9403152, 28.7734222, -61.7216187, 61.7451248
14: -23.5582600, 39.2390671, -23.5230694, 39.1645966, -60.0888596, 60.1092529
15: -18.9678383, 25.8919029, -18.9501915, 25.8452873, -44.8131256, 44.8420944
16: -32.7519493, 19.9164162, -32.7409058, 19.8644714, -52.6164207, 52.6573219
17: -17.8043766, 38.5050888, -17.7862167, 38.4351692, -55.2579460, 55.3073082
18: -25.8459511, 19.6262054, -25.7821884, 19.6227798, -45.4687309, 45.4083939
19: -26.4874821, 12.5135212, -26.4049473, 12.5120182, -38.9995003, 38.9184685
20: -21.1636925, 20.4625549, -21.0811310, 20.4570923, -41.6207848, 41.5436859
21: -25.7868099, 18.9118004, -25.6883640, 18.9088821, -44.6956940, 44.6001663
22: -22.1650448, 24.5442657, -22.0878487, 24.5429745, -46.7080193, 46.6321144
23: -21.7172432, 17.5157127, -21.6892967, 17.5055008, -39.2227440, 39.2050095
24: -32.2093582, 11.9117098, -32.1159821, 11.9115982, -44.1209564, 44.0276909
25: -18.1404343, 25.4336720, -18.0968857, 25.4348297, -43.5752640, 43.5305557
26: -29.2658806, 26.9726181, -29.2177620, 26.9724770, -56.2383575, 56.1903801
27: -32.1529503, 16.5573063, -32.0968552, 16.5571861, -47.8515511, 47.7817230
28: -21.5604820, 21.7160835, -21.5199070, 21.7136631, -43.2741470, 43.2359924
29: -23.7488213, 22.2372093, -23.6836147, 22.2375603, -45.9863815, 45.9208221
30: -29.6917915, 16.8771534, -29.6121998, 16.8673019, -46.0122719, 45.9416962
31: -26.4492207, 19.1118660, -26.3422432, 19.1077232, -45.5569458, 45.4541092
32: -42.2870865, 8.5082436, -42.2169113, 8.4945641, -47.6550369, 47.5792389
33: -72.4678116, -5.5824823, -72.3260803, -5.5842705, -61.4431305, 61.3220215
34: -56.5515938, -5.4618225, -56.4604187, -5.4666920, -43.6965256, 43.6125450
35: -50.2267532, 0.0670147, -50.1114655, 0.0694246, -48.3823395, 48.2744141
36: -47.8377495, 4.9744291, -47.7444458, 4.9724808, -52.1134262, 52.0222321
37: -83.7412262, -17.4286575, -83.6359558, -17.4323006, -58.5230560, 58.4544258
38: -58.7557220, 3.2664509, -58.6087646, 3.2657652, -61.4281082, 61.2825165
39: -79.0796280, -11.5620642, -78.9247742, -11.5624313, -65.4588623, 65.3074036
40: -67.7292938, -18.3100548, -67.6424026, -18.3143425, -41.2861710, 41.2214203
41: -55.2296371, -6.8051100, -55.1704941, -6.8166428, -42.3721237, 42.3226929
42: -33.9609337, 6.8302355, -33.9511795, 6.8211794, -37.7456207, 37.7033997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9830143, upper bound: 44.8703901
time: 41.53 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0392273, upper bound: 45.0392269
time: 44.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 87.99 seconds
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.8671733, upper bound: 45.0201639
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.8671733, upper bound: 45.0398400
IS_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.7533869, upper bound: 44.9830142
IS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.9055786, upper bound: 45.0392273
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.8642075, upper bound: 45.0201639
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.8642075, upper bound: 45.0398400
IS_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.9764213, upper bound: 44.5880704
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -45.0357106, upper bound: 44.7619647
IS_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.9780646, upper bound: 44.7130292
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -45.0370111, upper bound: 44.8870783
IS_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.9817017, upper bound: 44.7460528
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -45.0379675, upper bound: 44.9146438
IS_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 87.99
Output dim: 14, lower bound: -44.9830143, upper bound: 44.8703901
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 87.99
Output dim: 14, lower bound: -45.0392273, upper bound: 45.0392269

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -28.0285797, 16.9680214, -28.0693512, 17.0065994, -44.2464027, 44.2643623
1: -13.6166620, 16.9581299, -13.6513081, 17.0100632, -30.6267242, 30.6094379
2: -14.0097132, 21.5129776, -14.0614786, 21.5768890, -35.3394394, 35.3461952
3: -12.8444271, 23.3110619, -12.8838310, 23.3612289, -36.2056580, 36.1948929
4: -21.4883938, 18.3246498, -21.5457630, 18.3965187, -39.8849106, 39.8704147
5: -11.9427776, 22.7203884, -11.9836178, 22.7640190, -34.7067947, 34.7040062
6: -50.6428375, -3.6477394, -50.6662941, -3.6279750, -40.5000420, 40.4395638
7: -16.3267746, 18.3134003, -16.3723507, 18.3707619, -34.6975365, 34.6857529
8: -18.2238159, 21.1813984, -18.2767296, 21.2423706, -39.4661865, 39.4581299
9: -16.5953865, 23.1627178, -16.6647396, 23.2178802, -38.4809494, 38.4655151
10: -24.1343346, 38.3636360, -24.2295570, 38.4466400, -61.6001282, 61.5858192
11: -24.6917858, 17.5550575, -24.7362499, 17.5900574, -42.2818451, 42.2913055
12: -28.4857864, 19.9983845, -28.5884342, 20.0835228, -46.7207527, 46.7200851
13: -32.8683167, 28.7139225, -32.9095154, 28.7462196, -61.6145363, 61.6234360
14: -23.2928734, 39.1016464, -23.4025402, 39.1555710, -59.8464127, 59.8478165
15: -18.8635712, 25.7894535, -18.9027424, 25.8218746, -44.6854477, 44.6921959
16: -32.6540909, 19.8003273, -32.6968765, 19.8460350, -52.5001259, 52.4972038
17: -17.6434631, 38.3901978, -17.7121658, 38.4248657, -55.1063614, 55.1115341
18: -25.7031155, 19.5640507, -25.7607517, 19.5995674, -45.3026810, 45.3248024
19: -26.3496933, 12.4646301, -26.3816185, 12.4803467, -38.8300400, 38.8462486
20: -21.0053654, 20.3987999, -21.0510139, 20.4266300, -41.4319954, 41.4498138
21: -25.6026936, 18.8342381, -25.6515408, 18.8661690, -44.4688644, 44.4857788
22: -22.0272408, 24.4937897, -22.0604858, 24.5124798, -46.5397186, 46.5542755
23: -21.6434574, 17.4767303, -21.6689491, 17.4931297, -39.1365891, 39.1456795
24: -32.0665588, 11.8483219, -32.1016998, 11.8800354, -43.9465942, 43.9500198
25: -18.0396595, 25.3922253, -18.0703125, 25.4155006, -43.4551620, 43.4625397
26: -29.1275425, 26.8943882, -29.1794987, 26.9482460, -56.0757904, 56.0738869
27: -32.0305023, 16.4940548, -32.0777283, 16.5246296, -47.6626320, 47.7282867
28: -21.4629955, 21.6718578, -21.4964142, 21.6922932, -43.1552887, 43.1682739
29: -23.6086922, 22.1859074, -23.6530609, 22.2220020, -45.8306961, 45.8389664
30: -29.5461826, 16.8054352, -29.5846519, 16.8392582, -45.8481789, 45.8406067
31: -26.2784100, 19.0562534, -26.3133316, 19.0670509, -45.3454590, 45.3695831
32: -42.1396980, 8.4188242, -42.1875305, 8.4517298, -47.4654694, 47.4432564
33: -72.2791748, -5.7217255, -72.3121414, -5.6730881, -61.1620712, 61.1646652
34: -56.4262085, -5.5406961, -56.4488487, -5.5251780, -43.4963722, 43.5212250
35: -50.0768166, -0.0186567, -50.0984955, 0.0018139, -48.1209030, 48.1695099
36: -47.6873512, 4.9038486, -47.7201958, 4.9095554, -51.8930206, 51.9180222
37: -83.5894699, -17.4989548, -83.6169586, -17.4805374, -58.3132401, 58.2898636
38: -58.5386314, 3.1730499, -58.5828705, 3.1886568, -61.1397858, 61.1623688
39: -78.8755798, -11.6494703, -78.9051514, -11.6341124, -65.1823120, 65.2001953
40: -67.6105347, -18.3794556, -67.6271667, -18.3533859, -41.1452827, 41.0870819
41: -55.1420708, -6.8805981, -55.1599236, -6.8651876, -42.2420006, 42.1654739
42: -33.8970451, 6.7616186, -33.9303131, 6.8003922, -37.6533661, 37.6029739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7133143, upper bound: 44.9633456
time: 47.97 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8651556, upper bound: 45.0195588
time: 41.05 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -28.0285797, 16.9680214, -28.1562309, 17.0725517, -44.3171234, 44.3464432
1: -13.6166620, 16.9581299, -13.6877737, 17.0543480, -30.6710091, 30.6459045
2: -14.0097132, 21.5129776, -14.0969343, 21.6241570, -35.3914413, 35.3793983
3: -12.8444271, 23.3110619, -12.9103270, 23.3998394, -36.2442665, 36.2213898
4: -21.4883938, 18.3246498, -21.5784645, 18.4177284, -39.9061203, 39.9031143
5: -11.9427776, 22.7203884, -12.0188522, 22.7999344, -34.7427139, 34.7392426
6: -50.6428375, -3.6477394, -50.7735558, -3.5687571, -40.5598068, 40.5557480
7: -16.3267746, 18.3134003, -16.4090214, 18.4045582, -34.7313309, 34.7224197
8: -18.2238159, 21.1813984, -18.3354855, 21.3121071, -39.5359230, 39.5168839
9: -16.5953865, 23.1627178, -16.7041626, 23.2753925, -38.5375443, 38.5037308
10: -24.1343346, 38.3636360, -24.2805214, 38.5276489, -61.6870804, 61.6386528
11: -24.6917858, 17.5550575, -24.7705879, 17.5987930, -42.2905807, 42.3256454
12: -28.4857864, 19.9983845, -28.6070595, 20.1305256, -46.7636719, 46.7371407
13: -32.8683167, 28.7139225, -32.9330215, 28.7941036, -61.6624222, 61.6469421
14: -23.2928734, 39.1016464, -23.5012550, 39.2342453, -59.9268150, 59.9472961
15: -18.8635712, 25.7894535, -18.9533615, 25.8750916, -44.7386627, 44.7428131
16: -32.6540909, 19.8003273, -32.7350464, 19.9064827, -52.5605736, 52.5353737
17: -17.6434631, 38.3901978, -17.7679749, 38.4989891, -55.1805992, 55.1679192
18: -25.7031155, 19.5640507, -25.8309937, 19.6112289, -45.3143463, 45.3950424
19: -26.3496933, 12.4646301, -26.4706345, 12.5115395, -38.8612328, 38.9352646
20: -21.0053654, 20.3987999, -21.1411324, 20.4598618, -41.4652252, 41.5399323
21: -25.6026936, 18.8342381, -25.7588234, 18.9084873, -44.5111809, 44.5930634
22: -22.0272408, 24.4937897, -22.1443501, 24.5394592, -46.5667000, 46.6381378
23: -21.6434574, 17.4767303, -21.7052116, 17.5124931, -39.1559525, 39.1819420
24: -32.0665588, 11.8483219, -32.2012253, 11.8965044, -43.9630623, 44.0495453
25: -18.0396595, 25.3922253, -18.1242466, 25.4298325, -43.4694901, 43.5164719
26: -29.1275425, 26.8943882, -29.2316189, 26.9678154, -56.0953598, 56.1260071
27: -32.0305023, 16.4940548, -32.1417274, 16.5430946, -47.6887436, 47.7985649
28: -21.4629955, 21.6718578, -21.5442390, 21.7129822, -43.1759796, 43.2160950
29: -23.6086922, 22.1859074, -23.7210064, 22.2332726, -45.8419647, 45.9069138
30: -29.5461826, 16.8054352, -29.6711540, 16.8712139, -45.8724442, 45.9249916
31: -26.2784100, 19.0562534, -26.4325008, 19.1060410, -45.3844528, 45.4887543
32: -42.1396980, 8.4188242, -42.2622299, 8.4996405, -47.5128784, 47.5188293
33: -72.2791748, -5.7217255, -72.4623032, -5.6061487, -61.2292099, 61.3150482
34: -56.4262085, -5.5406961, -56.5451393, -5.4741459, -43.5492554, 43.6219330
35: -50.0768166, -0.0186567, -50.2202873, 0.0564013, -48.1763382, 48.2920074
36: -47.6873512, 4.9038486, -47.8208694, 4.9685650, -51.9519348, 52.0207367
37: -83.5894699, -17.4989548, -83.7299881, -17.4395962, -58.3535919, 58.4052849
38: -58.5386314, 3.1730499, -58.7415428, 3.2589159, -61.2105637, 61.3209839
39: -78.8755798, -11.6494703, -79.0694427, -11.5691299, -65.2474747, 65.3661270
40: -67.6105347, -18.3794556, -67.7210541, -18.3202705, -41.1769257, 41.1830368
41: -55.1420708, -6.8805981, -55.2216568, -6.8158684, -42.2928391, 42.2314911
42: -33.8970451, 6.7616186, -33.9440002, 6.8204136, -37.6753845, 37.6178284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7133143, upper bound: 44.9830142
time: 66.39 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8651556, upper bound: 45.0392273
time: 50.01 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -28.1143112, 17.0320702, -28.1240540, 17.0111122, -44.3299942, 44.3762474
1: -13.6523752, 17.0003414, -13.6829281, 17.0133858, -30.6657600, 30.6832695
2: -14.0444717, 21.5587273, -14.0924797, 21.5803375, -35.3748512, 35.4140396
3: -12.8703003, 23.3483391, -12.9100981, 23.3671303, -36.2374306, 36.2584381
4: -21.5202942, 18.3442745, -21.5734043, 18.4053936, -39.9256897, 39.9176788
5: -11.9776602, 22.7547588, -12.0174761, 22.7692318, -34.7468910, 34.7722359
6: -50.7492943, -3.5953155, -50.6696625, -3.5761037, -40.6675262, 40.5183411
7: -16.3622456, 18.3434486, -16.4061813, 18.3747406, -34.7369843, 34.7496300
8: -18.2813072, 21.2489967, -18.3343964, 21.2481480, -39.5294571, 39.5833931
9: -16.6314507, 23.2190437, -16.6985035, 23.2223625, -38.5126686, 38.5537643
10: -24.1829414, 38.4432983, -24.2683144, 38.4569702, -61.6425934, 61.7138786
11: -24.7242012, 17.5623016, -24.7418385, 17.6014175, -42.3256187, 42.3041382
12: -28.5012894, 20.0445824, -28.5926552, 20.1181717, -46.7563629, 46.7730179
13: -32.8837547, 28.7601624, -32.9216385, 28.7639961, -61.6477509, 61.6818008
14: -23.3875465, 39.1799278, -23.4694138, 39.1600952, -59.9216003, 59.9944344
15: -18.9059906, 25.8415298, -18.9402046, 25.8289795, -44.7349701, 44.7817345
16: -32.6908379, 19.8559361, -32.7227173, 19.8547935, -52.5456314, 52.5786514
17: -17.6961689, 38.4630508, -17.7508316, 38.4294891, -55.1455765, 55.2296448
18: -25.7719631, 19.5709381, -25.7680836, 19.6080017, -45.3799667, 45.3390198
19: -26.4378033, 12.4952326, -26.3896694, 12.5101681, -38.9479713, 38.8849030
20: -21.0939674, 20.4315281, -21.0596428, 20.4547920, -41.5487595, 41.4911728
21: -25.7085495, 18.8760014, -25.6626205, 18.9058132, -44.6143646, 44.5386200
22: -22.1046734, 24.5195274, -22.0690365, 24.5387287, -46.6434021, 46.5885620
23: -21.6787186, 17.4955463, -21.6775723, 17.5033855, -39.1821060, 39.1731186
24: -32.1649246, 11.8644018, -32.1086807, 11.8967018, -44.0616264, 43.9730835
25: -18.0909405, 25.4055805, -18.0812206, 25.4313660, -43.5223083, 43.4868011
26: -29.1738148, 26.9131851, -29.1896763, 26.9684658, -56.1422806, 56.1028595
27: -32.0929260, 16.5080814, -32.0864563, 16.5423450, -47.7782135, 47.7615318
28: -21.5097599, 21.6919975, -21.5045147, 21.7111511, -43.2209091, 43.1965103
29: -23.6727448, 22.1964207, -23.6585102, 22.2340488, -45.9067917, 45.8549309
30: -29.6307220, 16.8365345, -29.5922241, 16.8618984, -45.9372215, 45.8790321
31: -26.3965416, 19.0943089, -26.3268986, 19.1042213, -45.5007629, 45.4212074
32: -42.2131691, 8.4656134, -42.1927147, 8.4895887, -47.5764236, 47.5128517
33: -72.4283600, -5.6565933, -72.3214111, -5.6062508, -61.3782578, 61.2364960
34: -56.5217667, -5.4902554, -56.4544563, -5.4742002, -43.6462898, 43.5756721
35: -50.1948090, 0.0351048, -50.1054993, 0.0598602, -48.3131256, 48.2309113
36: -47.7823334, 4.9624386, -47.7273598, 4.9691448, -52.0490112, 51.9849014
37: -83.6980286, -17.4593277, -83.6249390, -17.4427032, -58.4651489, 58.3591309
38: -58.6957703, 3.2423611, -58.5951385, 3.2585707, -61.3660126, 61.2322311
39: -79.0389328, -11.5856495, -78.9155655, -11.5687551, -65.4127960, 65.2726212
40: -67.7029343, -18.3520031, -67.6348877, -18.3272285, -41.2636223, 41.1329041
41: -55.2029266, -6.8363781, -55.1630974, -6.8237143, -42.3503532, 42.2513580
42: -33.9100418, 6.7804193, -33.9343834, 6.8145142, -37.6845436, 37.6448898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7758971, upper bound: 45.0370108
time: 152.10 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7758971, upper bound: 44.8965129
time: 43.49 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -28.0770760, 17.0262833, -28.0754089, 17.0258694, -44.3149719, 44.3174744
1: -13.6547165, 17.0377121, -13.6539860, 17.0373230, -30.6920395, 30.6916981
2: -14.0662527, 21.6111450, -14.0657177, 21.6107864, -35.4304123, 35.4333649
3: -12.8874397, 23.3922691, -12.8868618, 23.3887939, -36.2762337, 36.2791290
4: -21.5535297, 18.4342957, -21.5527363, 18.4339695, -39.9874992, 39.9870300
5: -11.9871778, 22.7924099, -11.9865952, 22.7889538, -34.7761307, 34.7790070
6: -50.6773453, -3.6148510, -50.6774025, -3.6213083, -40.5433006, 40.5141602
7: -16.3769054, 18.4033203, -16.3760948, 18.4017944, -34.7787018, 34.7794151
8: -18.2843628, 21.2760792, -18.2832718, 21.2750969, -39.5594597, 39.5593491
9: -16.6996784, 23.2263870, -16.6981716, 23.2260838, -38.5744324, 38.5640678
10: -24.2791519, 38.4584503, -24.2769890, 38.4578667, -61.7339020, 61.7294426
11: -24.7605534, 17.5961418, -24.7594204, 17.5955601, -42.3561134, 42.3555603
12: -28.6437073, 20.0922279, -28.6430702, 20.0916195, -46.8655548, 46.8683281
13: -32.9246826, 28.7569008, -32.9281921, 28.7555923, -61.6802750, 61.6850929
14: -23.4595776, 39.1603928, -23.4562416, 39.1600723, -59.9860573, 59.9621429
15: -18.9171906, 25.8386803, -18.9127350, 25.8381863, -44.7553787, 44.7514153
16: -32.7138214, 19.8559761, -32.7150879, 19.8557205, -52.5695419, 52.5710640
17: -17.7485886, 38.4309616, -17.7475624, 38.4305077, -55.1982117, 55.1878738
18: -25.7757397, 19.6145744, -25.7748489, 19.6143684, -45.3901062, 45.3894234
19: -26.3984699, 12.4823284, -26.3968945, 12.4821835, -38.8806534, 38.8792229
20: -21.0735779, 20.4293156, -21.0724964, 20.4288883, -41.5024643, 41.5018120
21: -25.6795368, 18.8694763, -25.6772709, 18.8692627, -44.5487976, 44.5467453
22: -22.0811653, 24.5172749, -22.0792713, 24.5166969, -46.5978622, 46.5965462
23: -21.6809864, 17.4963188, -21.6806774, 17.4952488, -39.1762352, 39.1769943
24: -32.1098175, 11.8952475, -32.1090050, 11.8949432, -44.0047607, 44.0042534
25: -18.0865173, 25.4193325, -18.0859928, 25.4189587, -43.5054779, 43.5053253
26: -29.2137260, 26.9530411, -29.2075863, 26.9522781, -56.1660042, 56.1606293
27: -32.0889740, 16.5388374, -32.0881042, 16.5394573, -47.7340012, 47.7512665
28: -21.5126724, 21.6953621, -21.5118084, 21.6947937, -43.2074661, 43.2071686
29: -23.6808815, 22.2259083, -23.6781731, 22.2255077, -45.9063873, 45.9040833
30: -29.6053009, 16.8451939, -29.6046314, 16.8446903, -45.9104156, 45.9022522
31: -26.3300781, 19.0728741, -26.3286476, 19.0705566, -45.4006348, 45.4015198
32: -42.2123718, 8.4603176, -42.2117271, 8.4566851, -47.5425186, 47.5280647
33: -72.3175964, -5.6493187, -72.3167877, -5.6510744, -61.2259827, 61.2450867
34: -56.4553223, -5.5128241, -56.4548149, -5.5176592, -43.5457573, 43.5571213
35: -50.1049728, 0.0124388, -50.1044807, 0.0113487, -48.2013626, 48.2121201
36: -47.7370644, 4.9153728, -47.7372971, 4.9128761, -51.9511414, 51.9561844
37: -83.6281967, -17.4695854, -83.6280060, -17.4701595, -58.3690872, 58.4053650
38: -58.5970421, 3.1962442, -58.5964890, 3.1958704, -61.1989365, 61.1993637
39: -78.9153214, -11.6270199, -78.9143677, -11.6277657, -65.2273560, 65.2321472
40: -67.6354218, -18.3431606, -67.6346893, -18.3404999, -41.1662140, 41.1815491
41: -55.1679039, -6.8544378, -55.1673012, -6.8580942, -42.2627487, 42.2699165
42: -33.9472618, 6.8102007, -33.9471054, 6.8070536, -37.7136230, 37.6775398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9428633, upper bound: 44.8506135
time: 52.60 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9988213, upper bound: 45.0195585
time: 43.02 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -28.0770760, 17.0262833, -28.1623306, 17.0918121, -44.3856964, 44.3995209
1: -13.6547165, 17.0377121, -13.6904879, 17.0816116, -30.7363281, 30.7281990
2: -14.0662527, 21.6111450, -14.1011829, 21.6580467, -35.4824409, 35.4665604
3: -12.8874397, 23.3922691, -12.9133806, 23.4274044, -36.3148422, 36.3056488
4: -21.5535297, 18.4342957, -21.5854416, 18.4551735, -40.0087051, 40.0197372
5: -11.9871778, 22.7924099, -12.0218143, 22.8248920, -34.8120689, 34.8142242
6: -50.6773453, -3.6148510, -50.7846947, -3.5620942, -40.6030693, 40.6303406
7: -16.3769054, 18.4033203, -16.4127922, 18.4355927, -34.8125000, 34.8161125
8: -18.2843628, 21.2760792, -18.3420372, 21.3448563, -39.6292191, 39.6181183
9: -16.6996784, 23.2263870, -16.7375851, 23.2835960, -38.6310120, 38.6022682
10: -24.2791519, 38.4584503, -24.3279648, 38.5388908, -61.8208694, 61.7822800
11: -24.7605534, 17.5961418, -24.7937737, 17.6042919, -42.3648453, 42.3899155
12: -28.6437073, 20.0922279, -28.6617165, 20.1386070, -46.9085083, 46.8853569
13: -32.9246826, 28.7569008, -32.9517097, 28.8035507, -61.7282333, 61.7086105
14: -23.4595776, 39.1603928, -23.5549278, 39.2387657, -60.0664444, 60.0615959
15: -18.9171906, 25.8386803, -18.9633636, 25.8914070, -44.8085976, 44.8020439
16: -32.7138214, 19.8559761, -32.7532349, 19.9161549, -52.6299744, 52.6092110
17: -17.7485886, 38.4309616, -17.8033447, 38.5046310, -55.2724457, 55.2442589
18: -25.7757397, 19.6145744, -25.8450890, 19.6260262, -45.4017639, 45.4596634
19: -26.3984699, 12.4823284, -26.4859066, 12.5133724, -38.9118423, 38.9682350
20: -21.0735779, 20.4293156, -21.1626167, 20.4621124, -41.5356903, 41.5919342
21: -25.6795368, 18.8694763, -25.7845459, 18.9115486, -44.5910873, 44.6540222
22: -22.0811653, 24.5172749, -22.1631413, 24.5436783, -46.6248436, 46.6804161
23: -21.6809864, 17.4963188, -21.7169495, 17.5146465, -39.1956329, 39.2132683
24: -32.1098175, 11.8952475, -32.2085419, 11.9114113, -44.0212288, 44.1037903
25: -18.0865173, 25.4193325, -18.1399002, 25.4332962, -43.5198135, 43.5592346
26: -29.2137260, 26.9530411, -29.2596817, 26.9718399, -56.1855659, 56.2127228
27: -32.0889740, 16.5388374, -32.1520844, 16.5579338, -47.7601318, 47.8215332
28: -21.5126724, 21.6953621, -21.5596504, 21.7154808, -43.2281532, 43.2550125
29: -23.6808815, 22.2259083, -23.7460899, 22.2368164, -45.9176979, 45.9720001
30: -29.6053009, 16.8451939, -29.6911087, 16.8766365, -45.9346771, 45.9866409
31: -26.3300781, 19.0728741, -26.4478168, 19.1095600, -45.4396362, 45.5206909
32: -42.2123718, 8.4603176, -42.2864342, 8.5046062, -47.5899506, 47.6036682
33: -72.3175964, -5.6493187, -72.4669342, -5.5841780, -61.2930832, 61.3954697
34: -56.4553223, -5.5128241, -56.5511017, -5.4666653, -43.5986404, 43.6578293
35: -50.1049728, 0.0124388, -50.2262306, 0.0659552, -48.2568054, 48.3346252
36: -47.7370644, 4.9153728, -47.8379593, 4.9718771, -52.0100937, 52.0588913
37: -83.6281967, -17.4695854, -83.7410431, -17.4292164, -58.4094543, 58.5207710
38: -58.5970421, 3.1962442, -58.7551727, 3.2661114, -61.2697220, 61.3579712
39: -78.9153214, -11.6270199, -79.0786591, -11.5627966, -65.2924805, 65.3980789
40: -67.6354218, -18.3431606, -67.7285767, -18.3073959, -41.1978607, 41.2775230
41: -55.1679039, -6.8544378, -55.2290688, -6.8087883, -42.3136063, 42.3359489
42: -33.9472618, 6.8102007, -33.9607773, 6.8270912, -37.7356300, 37.6923981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9428633, upper bound: 44.8703901
time: 25.76 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9988213, upper bound: 45.0392269
time: 51.86 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -28.1551628, 17.0743580, -28.0830040, 16.9732723, -44.3405724, 44.3717079
1: -13.6874599, 17.0591049, -13.6513290, 16.9714279, -30.6588879, 30.7104340
2: -14.0967646, 21.6277294, -14.0395794, 21.5231056, -35.3753204, 35.4276276
3: -12.9099016, 23.3960190, -12.8694859, 23.2940140, -36.2039146, 36.2655029
4: -21.5785713, 18.4252510, -21.5172195, 18.3524742, -39.9310455, 39.9424706
5: -12.0187483, 22.8014164, -11.9818983, 22.7162056, -34.7349548, 34.7833138
6: -50.7753677, -3.5711970, -50.6534042, -3.6303344, -40.6206055, 40.5577087
7: -16.4075851, 18.4005394, -16.3532524, 18.3003216, -34.7079086, 34.7537918
8: -18.3335419, 21.3132248, -18.2736473, 21.1836891, -39.5172310, 39.5868721
9: -16.6949730, 23.2775612, -16.5967026, 23.1541615, -38.4987488, 38.5404587
10: -24.2748699, 38.5295143, -24.1492958, 38.3533440, -61.6269531, 61.6857338
11: -24.7705975, 17.5988941, -24.6985359, 17.5628433, -42.3334427, 42.2974319
12: -28.6055431, 20.1304150, -28.4796524, 20.0185184, -46.7477646, 46.7546959
13: -32.9144363, 28.7922020, -32.8288002, 28.6992302, -61.6136665, 61.6210022
14: -23.4723625, 39.2351837, -23.2622204, 39.0818024, -59.9180870, 59.8739204
15: -18.9560909, 25.8738213, -18.8906651, 25.8055553, -44.7616463, 44.7644882
16: -32.7220306, 19.9064636, -32.6453094, 19.7883434, -52.5103760, 52.5517731
17: -17.7637501, 38.4994087, -17.6616573, 38.3919830, -55.1729012, 55.1899872
18: -25.8311405, 19.6190567, -25.7036915, 19.5944157, -45.4255562, 45.3227463
19: -26.4714165, 12.5116606, -26.3529396, 12.4980202, -38.9694366, 38.8646011
20: -21.1409988, 20.4597664, -21.0078926, 20.4273319, -41.5683289, 41.4676590
21: -25.7581825, 18.9087982, -25.6009502, 18.8777924, -44.6359749, 44.5097504
22: -22.1453247, 24.5378838, -22.0179691, 24.5109749, -46.6562996, 46.5558548
23: -21.7006226, 17.5119915, -21.6331120, 17.4837685, -39.1843910, 39.1451035
24: -32.2005005, 11.8983727, -32.0602837, 11.8696537, -44.0701523, 43.9586563
25: -18.1216507, 25.4296741, -18.0365162, 25.4068432, -43.5284958, 43.4661903
26: -29.2323494, 26.9671745, -29.1058884, 26.9116039, -56.1439514, 56.0730629
27: -32.1414185, 16.5364265, -32.0178299, 16.4925499, -47.8249207, 47.6796875
28: -21.5457726, 21.7117062, -21.4694977, 21.6974754, -43.2432480, 43.1812057
29: -23.7263355, 22.2325172, -23.6150608, 22.2011566, -45.9274902, 45.8475800
30: -29.6764526, 16.8712082, -29.5640888, 16.8272591, -45.9496841, 45.8780441
31: -26.4321671, 19.1034355, -26.2729130, 19.0770683, -45.5092354, 45.3763504
32: -42.2700233, 8.4978952, -42.1652985, 8.4430227, -47.5675621, 47.5216522
33: -72.4611816, -5.6279659, -72.2558060, -5.7227039, -61.3139648, 61.2043304
34: -56.5432663, -5.4870739, -56.4002914, -5.5437508, -43.6169357, 43.5202827
35: -50.2185593, 0.0428381, -50.0556641, -0.0034218, -48.3022766, 48.1554413
36: -47.8142624, 4.9570704, -47.6601334, 4.9106064, -52.0213318, 51.9149246
37: -83.7292480, -17.4459362, -83.5802765, -17.4860191, -58.4723511, 58.3886986
38: -58.7417068, 3.2443762, -58.5342255, 3.1968212, -61.3194580, 61.1737061
39: -79.0686951, -11.5785980, -78.8658905, -11.6131411, -65.4068527, 65.2321091
40: -67.7214355, -18.3408051, -67.5886688, -18.4089184, -41.2040138, 41.1898270
41: -55.2220497, -6.8255234, -55.1226196, -6.8847589, -42.2950249, 42.2958717
42: -33.9555359, 6.8214054, -33.9465027, 6.7750120, -37.6508942, 37.6928902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9083226, upper bound: 44.5112549
time: 28.84 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0354515, upper bound: 44.7617053
time: 28.55 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -28.1585693, 17.0797329, -28.1015472, 16.9893188, -44.3578415, 44.3968430
1: -13.6889095, 17.0664787, -13.6639347, 16.9929981, -30.6819077, 30.7304134
2: -14.0986347, 21.6390343, -14.0619688, 21.5557880, -35.4022522, 35.4599075
3: -12.9112873, 23.4130630, -12.8948698, 23.3416348, -36.2529221, 36.3079338
4: -21.5818691, 18.4382706, -21.5423317, 18.3898087, -39.9716797, 39.9806023
5: -12.0204105, 22.8139744, -12.0038233, 22.7505283, -34.7709389, 34.8177986
6: -50.7783394, -3.5609956, -50.6646996, -3.6025224, -40.6740456, 40.5520401
7: -16.4102173, 18.4131126, -16.3783283, 18.3358078, -34.7460251, 34.7914429
8: -18.3383160, 21.3251266, -18.3013153, 21.2182121, -39.5565262, 39.6264420
9: -16.7150383, 23.2790394, -16.6560555, 23.1844501, -38.5476685, 38.5820694
10: -24.3107452, 38.5319862, -24.2542133, 38.4090424, -61.7196426, 61.7785301
11: -24.7860432, 17.6008472, -24.7425060, 17.5838184, -42.3698616, 42.3433533
12: -28.6223869, 20.1333447, -28.5280342, 20.0571747, -46.8030243, 46.8020706
13: -32.9232864, 28.7972031, -32.8567581, 28.7174950, -61.6407814, 61.6539612
14: -23.5086899, 39.2363510, -23.3689461, 39.1173782, -59.9909248, 59.9574203
15: -18.9607868, 25.8819695, -18.9106636, 25.8295689, -44.7903557, 44.7926331
16: -32.7425194, 19.9094429, -32.7077370, 19.8257790, -52.5682983, 52.6171799
17: -17.7739944, 38.5016098, -17.6920280, 38.4087181, -55.2008553, 55.2096519
18: -25.8362656, 19.6218872, -25.7307034, 19.6034584, -45.4397240, 45.3525925
19: -26.4798298, 12.5123119, -26.3780422, 12.5073462, -38.9871750, 38.8903542
20: -21.1476669, 20.4606819, -21.0290108, 20.4359226, -41.5835876, 41.4896927
21: -25.7709713, 18.9095402, -25.6386890, 18.8896828, -44.6606522, 44.5482292
22: -22.1502724, 24.5414162, -22.0349350, 24.5232143, -46.6734848, 46.5763512
23: -21.7102146, 17.5132713, -21.6608028, 17.4964828, -39.2066956, 39.1740723
24: -32.2041702, 11.8996868, -32.0730705, 11.8745584, -44.0787277, 43.9727554
25: -18.1334877, 25.4308662, -18.0711823, 25.4222679, -43.5557556, 43.5020485
26: -29.2406807, 26.9701233, -29.1319885, 26.9356422, -56.1763229, 56.1021118
27: -32.1452026, 16.5486279, -32.0473938, 16.5274849, -47.8437195, 47.7221642
28: -21.5518875, 21.7145672, -21.4881725, 21.7066135, -43.2584991, 43.2027397
29: -23.7309151, 22.2350655, -23.6295738, 22.2106113, -45.9415283, 45.8646393
30: -29.6869240, 16.8732948, -29.5952892, 16.8456039, -45.9807281, 45.9095879
31: -26.4406376, 19.1042366, -26.3010025, 19.0828915, -45.5235291, 45.4052391
32: -42.2736053, 8.5038567, -42.1755409, 8.4589548, -47.5971909, 47.5335159
33: -72.4638596, -5.6064339, -72.2843475, -5.6602392, -61.3711090, 61.2548447
34: -56.5458946, -5.4699869, -56.4255295, -5.4923344, -43.6628342, 43.5630646
35: -50.2209053, 0.0588036, -50.0770569, 0.0429716, -48.3499374, 48.2186890
36: -47.8175507, 4.9717474, -47.6793861, 4.9515162, -52.0619965, 51.9486694
37: -83.7328568, -17.4414902, -83.5949554, -17.4723682, -58.4981308, 58.4085999
38: -58.7455750, 3.2636280, -58.5717430, 3.2527771, -61.3663025, 61.2321930
39: -79.0743179, -11.5701447, -78.8936005, -11.5885258, -65.4352798, 65.2693024
40: -67.7240372, -18.3314877, -67.6092224, -18.3818874, -41.2336349, 41.2042694
41: -55.2242432, -6.8128643, -55.1429710, -6.8475370, -42.3293228, 42.3122406
42: -33.9579315, 6.8256016, -33.9505196, 6.7893972, -37.6980820, 37.6949310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9097639, upper bound: 44.6372592
time: 46.74 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0367521, upper bound: 44.8868190
time: 38.61 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -28.1605606, 17.0868320, -28.1104469, 17.0123806, -44.3733177, 44.4121819
1: -13.6897335, 17.0746307, -13.6722775, 17.0169907, -30.7067242, 30.7469082
2: -14.0998583, 21.6471138, -14.0736322, 21.5800362, -35.4259186, 35.4819183
3: -12.9125462, 23.4138508, -12.8871536, 23.3457222, -36.2582703, 36.3010025
4: -21.5829315, 18.4424953, -21.5544395, 18.4039078, -39.9868393, 39.9969330
5: -12.0207367, 22.8157616, -11.9981651, 22.7582626, -34.7789993, 34.8139267
6: -50.7816544, -3.5658298, -50.6687622, -3.6041865, -40.6780624, 40.5779152
7: -16.4109402, 18.4245491, -16.3836517, 18.3665085, -34.7774506, 34.8082008
8: -18.3383236, 21.3339195, -18.3120174, 21.2442131, -39.5825348, 39.6459351
9: -16.7190323, 23.2824173, -16.6692524, 23.1990623, -38.5659142, 38.6020699
10: -24.2942257, 38.5369568, -24.2084522, 38.4112167, -61.7032318, 61.7451897
11: -24.7794609, 17.6029263, -24.7191029, 17.5844994, -42.3639603, 42.3220291
12: -28.6454792, 20.1363029, -28.5957794, 20.0868568, -46.8564758, 46.8633232
13: -32.9393539, 28.7998123, -32.9042511, 28.7534943, -61.6928482, 61.7040634
14: -23.5219402, 39.2379456, -23.4122581, 39.1285706, -60.0154686, 59.9980736
15: -18.9631577, 25.8837509, -18.9217815, 25.8201370, -44.7832947, 44.8055344
16: -32.7314758, 19.9134617, -32.6770973, 19.8218880, -52.5533638, 52.5905609
17: -17.7941246, 38.5029030, -17.7527618, 38.4171257, -55.2285995, 55.2671471
18: -25.8408241, 19.6233788, -25.7537766, 19.6090031, -45.4498291, 45.3771553
19: -26.4790611, 12.5128708, -26.3789368, 12.5021133, -38.9811745, 38.8918076
20: -21.1570339, 20.4616451, -21.0584621, 20.4479637, -41.6049957, 41.5201073
21: -25.7740116, 18.9110336, -25.6492043, 18.8964691, -44.6704788, 44.5602379
22: -22.1600780, 24.5407028, -22.0643787, 24.5294495, -46.6895294, 46.6050797
23: -21.7076416, 17.5144005, -21.6605797, 17.4922180, -39.1998596, 39.1749802
24: -32.2056808, 11.9103928, -32.1020279, 11.9062824, -44.1119614, 44.0124207
25: -18.1286011, 25.4324741, -18.0596104, 25.4184151, -43.5470161, 43.4920845
26: -29.2575474, 26.9696579, -29.1859589, 26.9476566, -56.2052040, 56.1556168
27: -32.1491661, 16.5450974, -32.0656357, 16.5178223, -47.8338852, 47.7371941
28: -21.5543652, 21.7132378, -21.5001869, 21.7039509, -43.2583160, 43.2134247
29: -23.7442341, 22.2346611, -23.6651859, 22.2273197, -45.9715538, 45.8998489
30: -29.6813011, 16.8750610, -29.5790539, 16.8480854, -45.9801865, 45.8973274
31: -26.4407578, 19.1110420, -26.3131161, 19.1009960, -45.5417557, 45.4241562
32: -42.2835159, 8.5022917, -42.2054291, 8.4775372, -47.6438408, 47.5658112
33: -72.4651184, -5.6039762, -72.2965240, -5.6485291, -61.3808289, 61.2704620
34: -56.5489731, -5.4789057, -56.4344788, -5.5186815, -43.6495819, 43.5688820
35: -50.2243919, 0.0510740, -50.0861778, 0.0221777, -48.3337479, 48.2223892
36: -47.8344574, 4.9597282, -47.7194786, 4.9311142, -52.0736084, 51.9820862
37: -83.7376099, -17.4331131, -83.6168213, -17.4472809, -58.5173492, 58.4324875
38: -58.7518387, 3.2471390, -58.5697250, 3.2088919, -61.3681564, 61.2208176
39: -79.0740280, -11.5705214, -78.8960724, -11.5881786, -65.4275742, 65.2691727
40: -67.7266388, -18.3193684, -67.6203232, -18.3470421, -41.2624741, 41.2053947
41: -55.2274704, -6.8178186, -55.1491966, -6.8591290, -42.3709259, 42.3053207
42: -33.9585419, 6.8260241, -33.9465256, 6.8056059, -37.7145195, 37.7005234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8857963, upper bound: 44.9119827
time: 48.65 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8857963, upper bound: 44.7721864
time: 42.02 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -28.1639709, 17.0922165, -28.1289635, 17.0284576, -44.3905602, 44.4372902
1: -13.6911879, 17.0820007, -13.6848669, 17.0385742, -30.7297630, 30.7668686
2: -14.1017084, 21.6584167, -14.0959902, 21.6127243, -35.4528694, 35.5141525
3: -12.9139395, 23.4308968, -12.9125319, 23.3933582, -36.3072968, 36.3434296
4: -21.5862255, 18.4555130, -21.5795784, 18.4412575, -40.0274811, 40.0350914
5: -12.0224047, 22.8283195, -12.0200682, 22.7926445, -34.8150482, 34.8483887
6: -50.7846298, -3.5556145, -50.6799660, -3.5763912, -40.7314796, 40.5722656
7: -16.4135857, 18.4371357, -16.4087219, 18.4019928, -34.8155785, 34.8458557
8: -18.3431015, 21.3458290, -18.3397064, 21.2787342, -39.6218338, 39.6855354
9: -16.7390900, 23.2838955, -16.7285843, 23.2293587, -38.6147957, 38.6436577
10: -24.3301029, 38.5394440, -24.3134117, 38.4668884, -61.7958679, 61.8380013
11: -24.7949104, 17.6048737, -24.7630672, 17.6054649, -42.4003754, 42.3679428
12: -28.6623459, 20.1392345, -28.6441517, 20.1255054, -46.9117355, 46.9107399
13: -32.9481964, 28.8048096, -32.9321671, 28.7717934, -61.7199898, 61.7369766
14: -23.5582600, 39.2390671, -23.5190067, 39.1641464, -60.0883102, 60.0816040
15: -18.9678383, 25.8919029, -18.9418163, 25.8441124, -44.8119507, 44.8337173
16: -32.7519493, 19.9164162, -32.7395020, 19.8593102, -52.6112595, 52.6559181
17: -17.8043766, 38.5050888, -17.7831154, 38.4338531, -55.2565918, 55.2867737
18: -25.8459511, 19.6262054, -25.7807846, 19.6180782, -45.4640274, 45.4069901
19: -26.4874821, 12.5135212, -26.4040394, 12.5114479, -38.9989319, 38.9175606
20: -21.1636925, 20.4625549, -21.0796089, 20.4565659, -41.6202583, 41.5421638
21: -25.7868099, 18.9118004, -25.6869526, 18.9083538, -44.6951637, 44.5987549
22: -22.1650448, 24.5442657, -22.0813560, 24.5416813, -46.7067261, 46.6256218
23: -21.7172432, 17.5157127, -21.6883125, 17.5049210, -39.2221642, 39.2040253
24: -32.2093582, 11.9117098, -32.1147957, 11.9111958, -44.1205521, 44.0265045
25: -18.1404343, 25.4336720, -18.0942688, 25.4338436, -43.5742798, 43.5279388
26: -29.2658806, 26.9726181, -29.2121410, 26.9716988, -56.2375793, 56.1847610
27: -32.1529503, 16.5573063, -32.0951767, 16.5527687, -47.8526840, 47.7796783
28: -21.5604820, 21.7160835, -21.5188675, 21.7130775, -43.2735596, 43.2349510
29: -23.7488213, 22.2372093, -23.6797218, 22.2367706, -45.9855919, 45.9169312
30: -29.6917915, 16.8771534, -29.6102486, 16.8664207, -46.0112381, 45.9289017
31: -26.4492207, 19.1118660, -26.3411865, 19.1068192, -45.5560379, 45.4530525
32: -42.2870865, 8.5082436, -42.2156982, 8.4934607, -47.6735001, 47.5776939
33: -72.4678116, -5.5824823, -72.3250809, -5.5860367, -61.4380035, 61.3210144
34: -56.5515938, -5.4618225, -56.4596825, -5.4672832, -43.6954765, 43.6116562
35: -50.2267532, 0.0670147, -50.1075859, 0.0685730, -48.3814392, 48.2856636
36: -47.8377495, 4.9744291, -47.7387123, 4.9720573, -52.1143036, 52.0158844
37: -83.7412262, -17.4286575, -83.6314850, -17.4336414, -58.5431213, 58.4523849
38: -58.7557220, 3.2664509, -58.6072655, 3.2648420, -61.4149933, 61.2792969
39: -79.0796280, -11.5620642, -78.9237823, -11.5635853, -65.4560394, 65.3063507
40: -67.7292938, -18.3100548, -67.6408768, -18.3199940, -41.2920685, 41.2198372
41: -55.2296371, -6.8051100, -55.1695900, -6.8218660, -42.4052429, 42.3216743
42: -33.9609337, 6.8302355, -33.9505157, 6.8199806, -37.7617073, 37.7025757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8870786, upper bound: 45.0370105
time: 44.85 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8870786, upper bound: 44.8965125
time: 41.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 88.90 seconds
IS_A1_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.7133143, upper bound: 44.9633456
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.8651556, upper bound: 45.0195588
IS_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.7133143, upper bound: 44.9830142
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.8651556, upper bound: 45.0392273
IS_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.7758971, upper bound: 45.0370108
IS_A1_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.7758971, upper bound: 44.8965129
IS_A2_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.9428633, upper bound: 44.8506135
IS_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.9988213, upper bound: 45.0195585
IS_A2_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.9428633, upper bound: 44.8703901
IS_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.9988213, upper bound: 45.0392269
IS_A2_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.9083226, upper bound: 44.5112549
IS_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -45.0354515, upper bound: 44.7617053
IS_A2_A2_B1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.9097639, upper bound: 44.6372592
IS_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -45.0367521, upper bound: 44.8868190
IS_A2_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.8857963, upper bound: 44.9119827
IS_A2_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.8857963, upper bound: 44.7721864
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.8870786, upper bound: 45.0370105
IS_A2_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 88.90
Output dim: 14, lower bound: -44.8870786, upper bound: 44.8965125

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -28.0274200, 16.9661045, -28.0693512, 17.0065994, -44.2451973, 44.2551727
1: -13.6159210, 16.9560661, -13.6513081, 17.0100632, -30.6259842, 30.6073742
2: -14.0090065, 21.5114594, -14.0614786, 21.5768890, -35.3386917, 35.3325043
3: -12.8438101, 23.3097191, -12.8838310, 23.3612289, -36.2050400, 36.1935501
4: -21.4875793, 18.3230705, -21.5457630, 18.3965187, -39.8840981, 39.8688354
5: -11.9424505, 22.7188606, -11.9836178, 22.7640190, -34.7064705, 34.7024765
6: -50.6420135, -3.6545329, -50.6662941, -3.6279750, -40.4991875, 40.4593773
7: -16.3255730, 18.3096447, -16.3723507, 18.3707619, -34.6963348, 34.6819954
8: -18.2225723, 21.1792603, -18.2767296, 21.2423706, -39.4649429, 39.4559898
9: -16.5920372, 23.1614952, -16.6647396, 23.2178802, -38.4708214, 38.4640007
10: -24.1319523, 38.3623085, -24.2295570, 38.4466400, -61.5790100, 61.5842285
11: -24.6898460, 17.5535583, -24.7362499, 17.5900574, -42.2799034, 42.2898102
12: -28.4826488, 19.9976273, -28.5884342, 20.0835228, -46.7094383, 46.7193527
13: -32.8602409, 28.7122536, -32.9095154, 28.7462196, -61.6064606, 61.6217690
14: -23.2888336, 39.1012497, -23.4025402, 39.1555710, -59.8187447, 59.8472786
15: -18.8554001, 25.7883167, -18.9027424, 25.8218746, -44.6772766, 44.6910591
16: -32.6526833, 19.7954865, -32.6968765, 19.8460350, -52.4987183, 52.4923630
17: -17.6403809, 38.3889046, -17.7121658, 38.4248657, -55.0858383, 55.1102066
18: -25.7016907, 19.5592804, -25.7607517, 19.5995674, -45.3012581, 45.3200302
19: -26.3487873, 12.4640675, -26.3816185, 12.4803467, -38.8291321, 38.8456879
20: -21.0038624, 20.3982849, -21.0510139, 20.4266300, -41.4304924, 41.4492989
21: -25.6012764, 18.8336964, -25.6515408, 18.8661690, -44.4674454, 44.4852371
22: -22.0207863, 24.4925385, -22.0604858, 24.5124798, -46.5332642, 46.5530243
23: -21.6424675, 17.4761505, -21.6689491, 17.4931297, -39.1355972, 39.1450996
24: -32.0653725, 11.8479366, -32.1016998, 11.8800354, -43.9454079, 43.9496384
25: -18.0370197, 25.3912640, -18.0703125, 25.4155006, -43.4525223, 43.4615784
26: -29.1217480, 26.8936119, -29.1794987, 26.9482460, -56.0699921, 56.0731125
27: -32.0288544, 16.4896336, -32.0777283, 16.5246296, -47.6606407, 47.7310715
28: -21.4619560, 21.6712837, -21.4964142, 21.6922932, -43.1542511, 43.1676979
29: -23.6047993, 22.1851196, -23.6530609, 22.2220020, -45.8268013, 45.8381805
30: -29.5442390, 16.8045616, -29.5846519, 16.8392582, -45.8353729, 45.8395615
31: -26.2773514, 19.0553398, -26.3133316, 19.0670509, -45.3444023, 45.3686714
32: -42.1384850, 8.4177103, -42.1875305, 8.4517298, -47.4639282, 47.4616890
33: -72.2781754, -5.7235060, -72.3121414, -5.6730881, -61.1610641, 61.1595688
34: -56.4254646, -5.5412817, -56.4488487, -5.5251780, -43.4955063, 43.5202293
35: -50.0730171, -0.0194693, -50.0984955, 0.0018139, -48.1321564, 48.1686020
36: -47.6816406, 4.9033928, -47.7201958, 4.9095554, -51.8867340, 51.9188766
37: -83.5850143, -17.5002785, -83.6169586, -17.4805374, -58.3112106, 58.3100662
38: -58.5370560, 3.1721621, -58.5828705, 3.1886568, -61.1367950, 61.1490784
39: -78.8745880, -11.6505585, -78.9051514, -11.6341124, -65.1812897, 65.1973572
40: -67.6090393, -18.3851032, -67.6271667, -18.3533859, -41.1437187, 41.0930328
41: -55.1411667, -6.8857088, -55.1599236, -6.8651876, -42.2409859, 42.1985855
42: -33.8963776, 6.7603960, -33.9303131, 6.8003922, -37.6525345, 37.6190338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7561232, upper bound: 45.0173529
time: 29.26 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7561232, upper bound: 44.8768179
time: 46.32 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -28.0274200, 16.9661045, -28.1562309, 17.0725517, -44.3159332, 44.3372459
1: -13.6159210, 16.9560661, -13.6877737, 17.0543480, -30.6702690, 30.6438408
2: -14.0090065, 21.5114594, -14.0969343, 21.6241570, -35.3907013, 35.3657036
3: -12.8438101, 23.3097191, -12.9103270, 23.3998394, -36.2436485, 36.2200470
4: -21.4875793, 18.3230705, -21.5784645, 18.4177284, -39.9053078, 39.9015350
5: -11.9424505, 22.7188606, -12.0188522, 22.7999344, -34.7423859, 34.7377129
6: -50.6420135, -3.6545329, -50.7735558, -3.5687571, -40.5589676, 40.5755615
7: -16.3255730, 18.3096447, -16.4090214, 18.4045582, -34.7301331, 34.7186661
8: -18.2225723, 21.1792603, -18.3354855, 21.3121071, -39.5346794, 39.5147476
9: -16.5920372, 23.1614952, -16.7041626, 23.2753925, -38.5274124, 38.5022240
10: -24.1319523, 38.3623085, -24.2805214, 38.5276489, -61.6659622, 61.6370659
11: -24.6898460, 17.5535583, -24.7705879, 17.5987930, -42.2886391, 42.3241463
12: -28.4826488, 19.9976273, -28.6070595, 20.1305256, -46.7523499, 46.7364082
13: -32.8602409, 28.7122536, -32.9330215, 28.7941036, -61.6543427, 61.6452751
14: -23.2888336, 39.1012497, -23.5012550, 39.2342453, -59.8991508, 59.9467583
15: -18.8554001, 25.7883167, -18.9533615, 25.8750916, -44.7304916, 44.7416763
16: -32.6526833, 19.7954865, -32.7350464, 19.9064827, -52.5591660, 52.5305328
17: -17.6403809, 38.3889046, -17.7679749, 38.4989891, -55.1600609, 55.1665916
18: -25.7016907, 19.5592804, -25.8309937, 19.6112289, -45.3129196, 45.3902740
19: -26.3487873, 12.4640675, -26.4706345, 12.5115395, -38.8603287, 38.9347000
20: -21.0038624, 20.3982849, -21.1411324, 20.4598618, -41.4637222, 41.5394173
21: -25.6012764, 18.8336964, -25.7588234, 18.9084873, -44.5097656, 44.5925217
22: -22.0207863, 24.4925385, -22.1443501, 24.5394592, -46.5602455, 46.6368866
23: -21.6424675, 17.4761505, -21.7052116, 17.5124931, -39.1549606, 39.1813622
24: -32.0653725, 11.8479366, -32.2012253, 11.8965044, -43.9618759, 44.0491638
25: -18.0370197, 25.3912640, -18.1242466, 25.4298325, -43.4668503, 43.5155106
26: -29.1217480, 26.8936119, -29.2316189, 26.9678154, -56.0895615, 56.1252289
27: -32.0288544, 16.4896336, -32.1417274, 16.5430946, -47.6867523, 47.8013458
28: -21.4619560, 21.6712837, -21.5442390, 21.7129822, -43.1749382, 43.2155228
29: -23.6047993, 22.1851196, -23.7210064, 22.2332726, -45.8380737, 45.9061279
30: -29.5442390, 16.8045616, -29.6711540, 16.8712139, -45.8596268, 45.9239464
31: -26.2773514, 19.0553398, -26.4325008, 19.1060410, -45.3833923, 45.4878387
32: -42.1384850, 8.4177103, -42.2622299, 8.4996405, -47.5113373, 47.5372734
33: -72.2781754, -5.7235060, -72.4623032, -5.6061487, -61.2282028, 61.3099365
34: -56.4254646, -5.5412817, -56.5451393, -5.4741459, -43.5484009, 43.6209373
35: -50.0730171, -0.0194693, -50.2202873, 0.0564013, -48.1875992, 48.2910995
36: -47.6816406, 4.9033928, -47.8208694, 4.9685650, -51.9456482, 52.0216064
37: -83.5850143, -17.5002785, -83.7299881, -17.4395962, -58.3515625, 58.4254951
38: -58.5370560, 3.1721621, -58.7415428, 3.2589159, -61.2075729, 61.3076859
39: -78.8745880, -11.6505585, -79.0694427, -11.5691299, -65.2464523, 65.3632965
40: -67.6090393, -18.3851032, -67.7210541, -18.3202705, -41.1753845, 41.1890297
41: -55.1411667, -6.8857088, -55.2216568, -6.8158684, -42.2918396, 42.2646065
42: -33.8963776, 6.7603960, -33.9440002, 6.8204136, -37.6745453, 37.6338997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7355456, upper bound: 45.0370106
time: 23.42 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7355456, upper bound: 44.8965126
time: 31.71 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -28.0876656, 16.9930267, -28.1240540, 17.0111122, -44.3126717, 44.3369522
1: -13.6307449, 16.9538231, -13.6829281, 17.0133858, -30.6441307, 30.6367512
2: -14.0098085, 21.5011196, -14.0924797, 21.5803375, -35.3537216, 35.3528633
3: -12.8545685, 23.2955475, -12.9100981, 23.3671303, -36.2216988, 36.2056465
4: -21.4823647, 18.2922058, -21.5734043, 18.4053936, -39.8877563, 39.8656082
5: -11.9630833, 22.7124634, -12.0174761, 22.7692318, -34.7323151, 34.7299385
6: -50.7335434, -3.6225114, -50.6696625, -3.5761037, -40.6307678, 40.4856148
7: -16.3324966, 18.2767677, -16.4061813, 18.3747406, -34.7072372, 34.6829491
8: -18.2418938, 21.1872578, -18.3343964, 21.2481480, -39.4900436, 39.5216522
9: -16.5580063, 23.1738091, -16.6985035, 23.2223625, -38.4370689, 38.5299797
10: -24.1218033, 38.3853455, -24.2683144, 38.4569702, -61.5726776, 61.6748772
11: -24.7027645, 17.5395622, -24.7418385, 17.6014175, -42.3041840, 42.2814026
12: -28.3836536, 19.9756508, -28.5926552, 20.1181717, -46.6351700, 46.7215614
13: -32.8070526, 28.7048321, -32.9216385, 28.7639961, -61.5710487, 61.6264725
14: -23.2343597, 39.1330338, -23.4694138, 39.1600952, -59.7661209, 59.9785995
15: -18.8696785, 25.8258553, -18.9402046, 25.8289795, -44.6986580, 44.7660599
16: -32.6579399, 19.8210526, -32.7227173, 19.8547935, -52.5127335, 52.5437698
17: -17.6036453, 38.4371758, -17.7508316, 38.4294891, -55.0532684, 55.2206459
18: -25.7205048, 19.5559921, -25.7680836, 19.6080017, -45.3285065, 45.3240738
19: -26.4112434, 12.4907999, -26.3896694, 12.5101681, -38.9214096, 38.8804703
20: -21.0424156, 20.4107037, -21.0596428, 20.4547920, -41.4972076, 41.4703445
21: -25.6582813, 18.8570251, -25.6626205, 18.9058132, -44.5640945, 44.5196457
22: -22.0504017, 24.5006332, -22.0690365, 24.5387287, -46.5891304, 46.5696716
23: -21.6507530, 17.4860210, -21.6775723, 17.5033855, -39.1541367, 39.1635933
24: -32.1224060, 11.8274803, -32.1086807, 11.8967018, -44.0191078, 43.9361610
25: -18.0658169, 25.3941574, -18.0812206, 25.4313660, -43.4971848, 43.4753799
26: -29.0855999, 26.8819790, -29.1896763, 26.9684658, -56.0540657, 56.0716553
27: -32.0444260, 16.4822693, -32.0864563, 16.5423450, -47.7531357, 47.7359886
28: -21.4787979, 21.6849709, -21.5045147, 21.7111511, -43.1899490, 43.1894836
29: -23.6192627, 22.1715851, -23.6585102, 22.2340488, -45.8533096, 45.8300934
30: -29.6150818, 16.8149719, -29.5922241, 16.8618984, -45.9126701, 45.8648453
31: -26.3550358, 19.0678520, -26.3268986, 19.1042213, -45.4592590, 45.3947525
32: -42.1723175, 8.4304276, -42.1927147, 8.4895887, -47.5124893, 47.4720268
33: -72.3873978, -5.7317543, -72.3214111, -5.6062508, -61.3484192, 61.1603470
34: -56.4874306, -5.5195332, -56.4544563, -5.4742002, -43.6141319, 43.5404854
35: -50.1639557, 0.0084038, -50.1054993, 0.0598602, -48.2770844, 48.1807404
36: -47.7198792, 4.9417953, -47.7273598, 4.9691448, -51.9752426, 51.9617157
37: -83.6607513, -17.4987450, -83.6249390, -17.4427032, -58.4238586, 58.3260956
38: -58.6593094, 3.2291222, -58.5951385, 3.2585707, -61.3039093, 61.2090530
39: -79.0081024, -11.6110964, -78.9155655, -11.5687551, -65.3869629, 65.2497635
40: -67.6702271, -18.4156532, -67.6348877, -18.3272285, -41.2171249, 41.1077576
41: -55.1769867, -6.8644218, -55.1630974, -6.8237143, -42.2918129, 42.2217903
42: -33.9099121, 6.7490358, -33.9343834, 6.8145142, -37.6545372, 37.6074638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A1_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7159582, upper bound: 44.8651315
time: 51.47 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6166964, upper bound: 44.9601527
time: 70.57 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -28.0770760, 17.0262833, -28.0742378, 17.0239468, -44.3057747, 44.3162346
1: -13.6547165, 17.0377121, -13.6532459, 17.0352459, -30.6899624, 30.6909580
2: -14.0662527, 21.6111450, -14.0649872, 21.6092682, -35.4167175, 35.4326172
3: -12.8874397, 23.3922691, -12.8862476, 23.3874416, -36.2748795, 36.2785187
4: -21.5535297, 18.4342957, -21.5519180, 18.4323711, -39.9859009, 39.9862137
5: -11.9871778, 22.7924099, -11.9862232, 22.7874107, -34.7745895, 34.7786331
6: -50.6773453, -3.6148510, -50.6765671, -3.6282663, -40.5631409, 40.5132866
7: -16.3769054, 18.4033203, -16.3748703, 18.3980217, -34.7749252, 34.7781906
8: -18.2843628, 21.2760792, -18.2820358, 21.2729263, -39.5572891, 39.5581131
9: -16.6996784, 23.2263870, -16.6948204, 23.2248573, -38.5729256, 38.5539436
10: -24.2791519, 38.4584503, -24.2746277, 38.4565887, -61.7323303, 61.7083435
11: -24.7605534, 17.5961418, -24.7574730, 17.5941372, -42.3546906, 42.3536148
12: -28.6437073, 20.0922279, -28.6398888, 20.0908813, -46.8647995, 46.8570099
13: -32.9246826, 28.7569008, -32.9200630, 28.7539635, -61.6786461, 61.6769638
14: -23.4595776, 39.1603928, -23.4521694, 39.1596413, -59.9855118, 59.9344864
15: -18.9171906, 25.8386803, -18.9043350, 25.8370285, -44.7542191, 44.7430153
16: -32.7138214, 19.8559761, -32.7136765, 19.8505478, -52.5643692, 52.5696526
17: -17.7485886, 38.4309616, -17.7444534, 38.4292221, -55.1968689, 55.1673431
18: -25.7757397, 19.6145744, -25.7734413, 19.6096611, -45.3853989, 45.3880157
19: -26.3984699, 12.4823284, -26.3959961, 12.4816160, -38.8800850, 38.8783264
20: -21.0735779, 20.4293156, -21.0709877, 20.4283676, -41.5019455, 41.5003052
21: -25.6795368, 18.8694763, -25.6758442, 18.8687115, -44.5482483, 44.5453186
22: -22.0811653, 24.5172749, -22.0728016, 24.5154343, -46.5965996, 46.5900764
23: -21.6809864, 17.4963188, -21.6796913, 17.4946728, -39.1756592, 39.1760101
24: -32.1098175, 11.8952475, -32.1078033, 11.8945522, -44.0043716, 44.0030518
25: -18.0865173, 25.4193325, -18.0833702, 25.4179611, -43.5044785, 43.5027008
26: -29.2137260, 26.9530411, -29.2019348, 26.9514656, -56.1651917, 56.1549759
27: -32.0889740, 16.5388374, -32.0864449, 16.5350323, -47.7351151, 47.7492065
28: -21.5126724, 21.6953621, -21.5107651, 21.6942215, -43.2068939, 43.2061272
29: -23.6808815, 22.2259083, -23.6742439, 22.2247257, -45.9056091, 45.9001541
30: -29.6053009, 16.8451939, -29.6026897, 16.8437881, -45.9093552, 45.8894348
31: -26.3300781, 19.0728741, -26.3276081, 19.0696564, -45.3997345, 45.4004822
32: -42.2123718, 8.4603176, -42.2105103, 8.4555893, -47.5609856, 47.5265312
33: -72.3175964, -5.6493187, -72.3158188, -5.6528416, -61.2208252, 61.2440186
34: -56.4553223, -5.5128241, -56.4540749, -5.5182676, -43.5447006, 43.5562286
35: -50.1049728, 0.0124388, -50.1005936, 0.0105410, -48.2004623, 48.2233734
36: -47.7370644, 4.9153728, -47.7315521, 4.9124441, -51.9520569, 51.9498672
37: -83.6281967, -17.4695854, -83.6235275, -17.4714737, -58.3891373, 58.4033546
38: -58.5970421, 3.1962442, -58.5949554, 3.1949415, -61.1858063, 61.1961823
39: -78.9153214, -11.6270199, -78.9133606, -11.6288776, -65.2245331, 65.2311172
40: -67.6354218, -18.3431606, -67.6331711, -18.3461609, -41.1720963, 41.1799736
41: -55.1679039, -6.8544378, -55.1664047, -6.8633528, -42.2958870, 42.2688980
42: -33.9472618, 6.8102007, -33.9464569, 6.8058319, -37.7296944, 37.6767082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8673439, upper bound: 45.0173521
time: 56.89 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8673439, upper bound: 44.7074438
time: 211.64 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -28.0770760, 17.0262833, -28.1611366, 17.0898991, -44.3765068, 44.3982925
1: -13.6547165, 17.0377121, -13.6897049, 17.0795383, -30.7342548, 30.7274170
2: -14.0662527, 21.6111450, -14.1004677, 21.6565361, -35.4687576, 35.4658012
3: -12.8874397, 23.3922691, -12.9127579, 23.4260559, -36.3134956, 36.3050270
4: -21.5535297, 18.4342957, -21.5846252, 18.4535904, -40.0071182, 40.0189209
5: -11.9871778, 22.7924099, -12.0214529, 22.8233585, -34.8105354, 34.8138618
6: -50.6773453, -3.6148510, -50.7838364, -3.5690541, -40.6229172, 40.6294708
7: -16.3769054, 18.4033203, -16.4115601, 18.4318314, -34.8087387, 34.8148804
8: -18.2843628, 21.2760792, -18.3407898, 21.3426857, -39.6270485, 39.6168671
9: -16.6996784, 23.2263870, -16.7342491, 23.2823677, -38.6295204, 38.5921402
10: -24.2791519, 38.4584503, -24.3256416, 38.5375595, -61.8192902, 61.7611732
11: -24.7605534, 17.5961418, -24.7918167, 17.6028652, -42.3634186, 42.3879585
12: -28.6437073, 20.0922279, -28.6585541, 20.1378765, -46.9077301, 46.8740578
13: -32.9246826, 28.7569008, -32.9435730, 28.8018494, -61.7265320, 61.7004738
14: -23.4595776, 39.1603928, -23.5508823, 39.2383423, -60.0659142, 60.0339470
15: -18.9171906, 25.8386803, -18.9549694, 25.8902664, -44.8074570, 44.7936478
16: -32.7138214, 19.8559761, -32.7518539, 19.9109783, -52.6248016, 52.6078300
17: -17.7485886, 38.4309616, -17.8002357, 38.5033340, -55.2711029, 55.2237625
18: -25.7757397, 19.6145744, -25.8436661, 19.6213226, -45.3970642, 45.4582405
19: -26.3984699, 12.4823284, -26.4850101, 12.5128040, -38.9112740, 38.9673386
20: -21.0735779, 20.4293156, -21.1611004, 20.4616165, -41.5351944, 41.5904160
21: -25.6795368, 18.8694763, -25.7831211, 18.9110336, -44.5905685, 44.6525955
22: -22.0811653, 24.5172749, -22.1566925, 24.5424118, -46.6235771, 46.6739655
23: -21.6809864, 17.4963188, -21.7159595, 17.5140495, -39.1950378, 39.2122803
24: -32.1098175, 11.8952475, -32.2073441, 11.9110222, -44.0208397, 44.1025925
25: -18.0865173, 25.4193325, -18.1372814, 25.4322643, -43.5187836, 43.5566139
26: -29.2137260, 26.9530411, -29.2541142, 26.9710503, -56.1847763, 56.2071533
27: -32.0889740, 16.5388374, -32.1504135, 16.5534916, -47.7612457, 47.8194923
28: -21.5126724, 21.6953621, -21.5586014, 21.7149010, -43.2275734, 43.2539635
29: -23.6808815, 22.2259083, -23.7422009, 22.2360249, -45.9169083, 45.9681091
30: -29.6053009, 16.8451939, -29.6891785, 16.8757534, -45.9336205, 45.9738197
31: -26.3300781, 19.0728741, -26.4467564, 19.1086235, -45.4387016, 45.5196304
32: -42.2123718, 8.4603176, -42.2851982, 8.5035000, -47.6084175, 47.6021385
33: -72.3175964, -5.6493187, -72.4659882, -5.5859470, -61.2879562, 61.3944397
34: -56.4553223, -5.5128241, -56.5503845, -5.4672604, -43.5975952, 43.6569519
35: -50.1049728, 0.0124388, -50.2223511, 0.0651226, -48.2558899, 48.3458557
36: -47.7370644, 4.9153728, -47.8322105, 4.9714508, -52.0109406, 52.0525589
37: -83.6281967, -17.4695854, -83.7365265, -17.4305592, -58.4294739, 58.5187759
38: -58.5970421, 3.1962442, -58.7536583, 3.2652140, -61.2565613, 61.3547974
39: -78.9153214, -11.6270199, -79.0776978, -11.5639372, -65.2896347, 65.3970871
40: -67.6354218, -18.3431606, -67.7270432, -18.3130550, -41.2037277, 41.2759628
41: -55.1679039, -6.8544378, -55.2281570, -6.8140354, -42.3467216, 42.3349266
42: -33.9472618, 6.8102007, -33.9601212, 6.8258657, -37.7517090, 37.6915665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8466622, upper bound: 45.0370106
time: 30.74 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8466622, upper bound: 44.8965122
time: 44.38 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -28.1551628, 17.0743580, -28.0812569, 16.9720268, -44.3360367, 44.3696671
1: -13.6874599, 17.0591049, -13.6490784, 16.9705620, -30.6580219, 30.7081833
2: -14.0967646, 21.6277294, -14.0375061, 21.5222664, -35.3744011, 35.4237022
3: -12.9099016, 23.3960190, -12.8673725, 23.2929668, -36.2028694, 36.2633896
4: -21.5785713, 18.4252510, -21.5145035, 18.3517876, -39.9303589, 39.9397545
5: -12.0187483, 22.8014164, -11.9788685, 22.7149315, -34.7336807, 34.7802849
6: -50.7753677, -3.5711970, -50.6486893, -3.6325603, -40.6185684, 40.5542679
7: -16.4075851, 18.4005394, -16.3491478, 18.2987289, -34.7063141, 34.7496872
8: -18.3335419, 21.3132248, -18.2694473, 21.1827126, -39.5162544, 39.5826721
9: -16.6949730, 23.2775612, -16.5953865, 23.1526260, -38.4939346, 38.5391273
10: -24.2748699, 38.5295143, -24.1480293, 38.3508797, -61.6064911, 61.6844749
11: -24.7705975, 17.5988941, -24.6935539, 17.5607414, -42.3313370, 42.2924500
12: -28.6055431, 20.1304150, -28.4782372, 20.0147858, -46.7476120, 46.7529640
13: -32.9144363, 28.7922020, -32.8273544, 28.6976318, -61.6120682, 61.6195564
14: -23.4723625, 39.2351837, -23.2590790, 39.0808868, -59.9170609, 59.8628540
15: -18.9560909, 25.8738213, -18.8880463, 25.8013878, -44.7574768, 44.7618675
16: -32.7220306, 19.9064636, -32.6436462, 19.7862091, -52.5082397, 52.5501099
17: -17.7637501, 38.4994087, -17.6591644, 38.3908310, -55.1715393, 55.1822624
18: -25.8311405, 19.6190567, -25.6991463, 19.5933647, -45.4245071, 45.3182030
19: -26.4714165, 12.5116606, -26.3518677, 12.4962931, -38.9677086, 38.8635292
20: -21.1409988, 20.4597664, -21.0062160, 20.4266968, -41.5676956, 41.4659805
21: -25.7581825, 18.9087982, -25.5989590, 18.8768654, -44.6350479, 44.5077591
22: -22.1453247, 24.5378838, -22.0153828, 24.5084267, -46.6537514, 46.5532684
23: -21.7006226, 17.5119915, -21.6322784, 17.4825821, -39.1832047, 39.1442719
24: -32.2005005, 11.8983727, -32.0580406, 11.8687820, -44.0692825, 43.9564133
25: -18.1216507, 25.4296741, -18.0348740, 25.4051476, -43.5267982, 43.4645462
26: -29.2323494, 26.9671745, -29.1014328, 26.9098549, -56.1422043, 56.0686073
27: -32.1414185, 16.5364265, -32.0110016, 16.4917221, -47.8239212, 47.6867294
28: -21.5457726, 21.7117062, -21.4682083, 21.6960449, -43.2418175, 43.1799164
29: -23.7263355, 22.2325172, -23.6131802, 22.1995983, -45.9259338, 45.8456955
30: -29.6764526, 16.8712082, -29.5621529, 16.8261509, -45.9482651, 45.8778114
31: -26.4321671, 19.1034355, -26.2712421, 19.0758286, -45.5079956, 45.3746796
32: -42.2700233, 8.4978952, -42.1621552, 8.4415865, -47.5711136, 47.5189590
33: -72.4611816, -5.6279659, -72.2536011, -5.7249546, -61.3056946, 61.2021027
34: -56.5432663, -5.4870739, -56.3990822, -5.5469408, -43.6038055, 43.5178337
35: -50.2185593, 0.0428381, -50.0533371, -0.0080938, -48.2836304, 48.1532364
36: -47.8142624, 4.9570704, -47.6583557, 4.9092417, -52.0235748, 51.9129105
37: -83.7292480, -17.4459362, -83.5789948, -17.4891720, -58.4208603, 58.3865700
38: -58.7417068, 3.2443762, -58.5299568, 3.1956654, -61.3254623, 61.1704636
39: -79.0686951, -11.5785980, -78.8635406, -11.6168833, -65.4084702, 65.2297745
40: -67.7214355, -18.3408051, -67.5873642, -18.4106941, -41.1796074, 41.1883736
41: -55.2220497, -6.8255234, -55.1217422, -6.8876371, -42.2707672, 42.2948990
42: -33.9555359, 6.8214054, -33.9458046, 6.7721291, -37.6362495, 37.6913376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8634271, upper bound: 44.7013961
time: 26.94 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8634271, upper bound: 44.6851322
time: 57.54 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -28.1585693, 17.0797329, -28.0997963, 16.9880810, -44.3532753, 44.3948021
1: -13.6889095, 17.0664787, -13.6616478, 16.9921360, -30.6810455, 30.7281265
2: -14.0986347, 21.6390343, -14.0599146, 21.5549316, -35.4013519, 35.4559746
3: -12.9112873, 23.4130630, -12.8927441, 23.3405800, -36.2518692, 36.3058090
4: -21.5818691, 18.4382706, -21.5396385, 18.3891296, -39.9710007, 39.9779091
5: -12.0204105, 22.8139744, -12.0007877, 22.7492580, -34.7696686, 34.8147621
6: -50.7783394, -3.5609956, -50.6599655, -3.6047702, -40.6720161, 40.5486031
7: -16.4102173, 18.4131126, -16.3742332, 18.3341980, -34.7444153, 34.7873459
8: -18.3383160, 21.3251266, -18.2971382, 21.2172184, -39.5555344, 39.6222649
9: -16.7150383, 23.2790394, -16.6547318, 23.1829147, -38.5428543, 38.5807381
10: -24.3107452, 38.5319862, -24.2529659, 38.4065666, -61.6991882, 61.7772446
11: -24.7860432, 17.6008472, -24.7375069, 17.5817127, -42.3677559, 42.3383560
12: -28.6223869, 20.1333447, -28.5265999, 20.0534477, -46.8028755, 46.8003464
13: -32.9232864, 28.7972031, -32.8553009, 28.7159100, -61.6391983, 61.6525040
14: -23.5086899, 39.2363510, -23.3657703, 39.1164932, -59.9898834, 59.9463501
15: -18.9607868, 25.8819695, -18.9080429, 25.8253746, -44.7861633, 44.7900124
16: -32.7425194, 19.9094429, -32.7060547, 19.8236485, -52.5661697, 52.6154976
17: -17.7739944, 38.5016098, -17.6895218, 38.4075470, -55.1995010, 55.2019119
18: -25.8362656, 19.6218872, -25.7261887, 19.6024036, -45.4386673, 45.3480759
19: -26.4798298, 12.5123119, -26.3769684, 12.5056324, -38.9854622, 38.8892822
20: -21.1476669, 20.4606819, -21.0273476, 20.4352837, -41.5829506, 41.4880295
21: -25.7709713, 18.9095402, -25.6366730, 18.8887787, -44.6597519, 44.5462112
22: -22.1502724, 24.5414162, -22.0323124, 24.5206909, -46.6709633, 46.5737305
23: -21.7102146, 17.5132713, -21.6599636, 17.4952774, -39.2054901, 39.1732330
24: -32.2041702, 11.8996868, -32.0708008, 11.8737144, -44.0778847, 43.9704895
25: -18.1334877, 25.4308662, -18.0695496, 25.4205837, -43.5540695, 43.5004158
26: -29.2406807, 26.9701233, -29.1275024, 26.9338837, -56.1745644, 56.0976257
27: -32.1452026, 16.5486279, -32.0405426, 16.5266533, -47.8427048, 47.7292137
28: -21.5518875, 21.7145672, -21.4868660, 21.7051506, -43.2570381, 43.2014313
29: -23.7309151, 22.2350655, -23.6276703, 22.2090645, -45.9399796, 45.8627357
30: -29.6869240, 16.8732948, -29.5933437, 16.8444920, -45.9793205, 45.9093475
31: -26.4406376, 19.1042366, -26.2993240, 19.0816269, -45.5222626, 45.4035606
32: -42.2736053, 8.5038567, -42.1724205, 8.4575024, -47.6007462, 47.5308113
33: -72.4638596, -5.6064339, -72.2821503, -5.6624756, -61.3628540, 61.2526321
34: -56.5458946, -5.4699869, -56.4243279, -5.4955578, -43.6497078, 43.5606117
35: -50.2209053, 0.0588036, -50.0747414, 0.0382929, -48.3313217, 48.2165031
36: -47.8175507, 4.9717474, -47.6776047, 4.9501734, -52.0642548, 51.9466705
37: -83.7328568, -17.4414902, -83.5936432, -17.4755554, -58.4466400, 58.4064789
38: -58.7455750, 3.2636280, -58.5674858, 3.2516069, -61.3723297, 61.2289047
39: -79.0743179, -11.5701447, -78.8912964, -11.5922565, -65.4368973, 65.2669830
40: -67.7240372, -18.3314877, -67.6079102, -18.3836823, -41.2092209, 41.2028236
41: -55.2242432, -6.8128643, -55.1421165, -6.8504152, -42.3050613, 42.3112640
42: -33.9579315, 6.8256016, -33.9498062, 6.7864866, -37.6834373, 37.6933746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=208, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8648726, upper bound: 44.8268415
time: 63.77 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8648726, upper bound: 44.8096277
time: 49.60 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -28.1347466, 17.0522079, -28.1289635, 17.0284576, -44.3722610, 44.3977699
1: -13.6694365, 17.0350819, -13.6848669, 17.0385742, -30.7080116, 30.7199478
2: -14.0670166, 21.6002350, -14.0959902, 21.6127243, -35.4316635, 35.4523201
3: -12.8957214, 23.3764095, -12.9125319, 23.3933582, -36.2890778, 36.2889404
4: -21.5482101, 18.4031448, -21.5795784, 18.4412575, -39.9894676, 39.9827232
5: -12.0055323, 22.7845440, -12.0200682, 22.7926445, -34.7981758, 34.8046112
6: -50.7683945, -3.5828376, -50.6799660, -3.5763912, -40.6941910, 40.5396500
7: -16.3823357, 18.3670940, -16.4087219, 18.4019928, -34.7843285, 34.7758179
8: -18.3036690, 21.2836533, -18.3397064, 21.2787342, -39.5824051, 39.6233597
9: -16.6651726, 23.2384796, -16.7285843, 23.2293587, -38.5385551, 38.6196938
10: -24.2688007, 38.4806976, -24.3134117, 38.4668884, -61.7256317, 61.7984352
11: -24.7731915, 17.5823479, -24.7630672, 17.6054649, -42.3786545, 42.3454132
12: -28.5443249, 20.0701656, -28.6441517, 20.1255054, -46.7901306, 46.8591118
13: -32.8711128, 28.7492409, -32.9321671, 28.7717934, -61.6429062, 61.6814079
14: -23.4043140, 39.1918983, -23.5190067, 39.1641464, -59.9320602, 60.0654564
15: -18.9314804, 25.8759861, -18.9418163, 25.8441124, -44.7755928, 44.8178024
16: -32.7186623, 19.8816185, -32.7395020, 19.8593102, -52.5779724, 52.6211205
17: -17.7111931, 38.4791908, -17.7831154, 38.4338531, -55.1636658, 55.2776833
18: -25.7943134, 19.6110573, -25.7807846, 19.6180782, -45.4123917, 45.3918419
19: -26.4605904, 12.5090580, -26.4040394, 12.5114479, -38.9720383, 38.9130974
20: -21.1119003, 20.4415112, -21.0796089, 20.4565659, -41.5684662, 41.5211182
21: -25.7361240, 18.8927555, -25.6869526, 18.9083538, -44.6444778, 44.5797081
22: -22.1100368, 24.5253296, -22.0813560, 24.5416813, -46.6517181, 46.6066856
23: -21.6890640, 17.5060959, -21.6883125, 17.5049210, -39.1939850, 39.1944084
24: -32.1667671, 11.8745060, -32.1147957, 11.9111958, -44.0779648, 43.9893036
25: -18.1150856, 25.4216385, -18.0942688, 25.4338436, -43.5489273, 43.5159073
26: -29.1768627, 26.9357491, -29.2121410, 26.9716988, -56.1485596, 56.1478882
27: -32.1038666, 16.5313931, -32.0951767, 16.5527687, -47.8270035, 47.7539024
28: -21.5290031, 21.7090397, -21.5188675, 21.7130775, -43.2420807, 43.2279053
29: -23.6935444, 22.2106876, -23.6797218, 22.2367706, -45.9303131, 45.8904114
30: -29.6759434, 16.8555431, -29.6102486, 16.8664207, -45.9864273, 45.9146576
31: -26.4076157, 19.0852757, -26.3411865, 19.1068192, -45.5144348, 45.4264603
32: -42.2455444, 8.4730072, -42.2156982, 8.4934607, -47.6088753, 47.5368423
33: -72.4266205, -5.6581182, -72.3250809, -5.5860367, -61.4079742, 61.2442169
34: -56.5168304, -5.4909849, -56.4596825, -5.4672832, -43.6627846, 43.5762215
35: -50.1955719, 0.0402050, -50.1075859, 0.0685730, -48.3450623, 48.2353210
36: -47.7742195, 4.9531946, -47.7387123, 4.9720573, -52.0386200, 51.9921036
37: -83.7036896, -17.4683170, -83.6314850, -17.4336414, -58.5012054, 58.4191513
38: -58.7189598, 3.2536907, -58.6072655, 3.2648420, -61.3524628, 61.2558670
39: -79.0487518, -11.5878553, -78.9237823, -11.5635853, -65.4301987, 65.2831116
40: -67.6964111, -18.3740673, -67.6408768, -18.3199940, -41.2451591, 41.1952248
41: -55.2022896, -6.8335876, -55.1695900, -6.8218660, -42.3457680, 42.2921371
42: -33.9605370, 6.7988586, -33.9505157, 6.8199806, -37.7314034, 37.6651192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6372594, upper bound: 44.9097637
time: 51.77 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8868192, upper bound: 45.0367514
time: 23.79 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 78.04 seconds
IS_A1_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.7561232, upper bound: 45.0173529
IS_A1_A1_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.7561232, upper bound: 44.8768179
IS_A1_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.7355456, upper bound: 45.0370106
IS_A1_A1_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.7355456, upper bound: 44.8965126
IS_A1_A2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.7159582, upper bound: 44.8651315
IS_A1_A2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.6166964, upper bound: 44.9601527
IS_A2_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8673439, upper bound: 45.0173521
IS_A2_A1_B2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8673439, upper bound: 44.7074438
IS_A2_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8466622, upper bound: 45.0370106
IS_A2_A1_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8466622, upper bound: 44.8965122
IS_A2_A2_B1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8634271, upper bound: 44.7013961
IS_A2_A2_B1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8634271, upper bound: 44.6851322
IS_A2_A2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8648726, upper bound: 44.8268415
IS_A2_A2_B1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8648726, upper bound: 44.8096277
IS_A2_A2_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.6372594, upper bound: 44.9097637
IS_A2_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 78.04
Output dim: 14, lower bound: -44.8868192, upper bound: 45.0367514

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -28.0007820, 16.9270573, -28.0693512, 17.0065994, -44.2278824, 44.2158813
1: -13.5942965, 16.9095383, -13.6513081, 17.0100632, -30.6043587, 30.5608463
2: -13.9743519, 21.4538574, -14.0614786, 21.5768890, -35.3175507, 35.2713127
3: -12.8280888, 23.2569008, -12.8838310, 23.3612289, -36.1893158, 36.1407318
4: -21.4496460, 18.2709942, -21.5457630, 18.3965187, -39.8461647, 39.8167572
5: -11.9278736, 22.6765556, -11.9836178, 22.7640190, -34.6918945, 34.6601715
6: -50.6262665, -3.6817207, -50.6662941, -3.6279750, -40.4624214, 40.4266701
7: -16.2958202, 18.2429771, -16.3723507, 18.3707619, -34.6665802, 34.6153259
8: -18.1831627, 21.1175041, -18.2767296, 21.2423706, -39.4255333, 39.3942337
9: -16.5186024, 23.1162872, -16.6647396, 23.2178802, -38.3952293, 38.4401970
10: -24.0708351, 38.3043518, -24.2295570, 38.4466400, -61.5090790, 61.5452080
11: -24.6684113, 17.5308418, -24.7362499, 17.5900574, -42.2584686, 42.2670898
12: -28.3650246, 19.9287186, -28.5884342, 20.0835228, -46.5882530, 46.6678925
13: -32.7835159, 28.6569901, -32.9095154, 28.7462196, -61.5297356, 61.5665054
14: -23.1356182, 39.0543289, -23.4025402, 39.1555710, -59.6632423, 59.8314362
15: -18.8191280, 25.7726421, -18.9027424, 25.8218746, -44.6410027, 44.6753845
16: -32.6197929, 19.7606316, -32.6968765, 19.8460350, -52.4658279, 52.4575081
17: -17.5478287, 38.3630447, -17.7121658, 38.4248657, -54.9935455, 55.1012039
18: -25.6502075, 19.5443535, -25.7607517, 19.5995674, -45.2497749, 45.3051071
19: -26.3222084, 12.4596004, -26.3816185, 12.4803467, -38.8025551, 38.8412170
20: -20.9522972, 20.3774815, -21.0510139, 20.4266300, -41.3789291, 41.4284973
21: -25.5509911, 18.8147488, -25.6515408, 18.8661690, -44.4171600, 44.4662895
22: -21.9665222, 24.4736214, -22.0604858, 24.5124798, -46.4790039, 46.5341072
23: -21.6144714, 17.4666328, -21.6689491, 17.4931297, -39.1076012, 39.1355820
24: -32.0228539, 11.8110342, -32.1016998, 11.8800354, -43.9028893, 43.9127350
25: -18.0118599, 25.3798065, -18.0703125, 25.4155006, -43.4273605, 43.4501190
26: -29.0335388, 26.8623962, -29.1794987, 26.9482460, -55.9817848, 56.0418930
27: -31.9803104, 16.4638157, -32.0777283, 16.5246296, -47.6355515, 47.7055244
28: -21.4309540, 21.6642895, -21.4964142, 21.6922932, -43.1232452, 43.1607056
29: -23.5513344, 22.1602783, -23.6530609, 22.2220020, -45.7733383, 45.8133392
30: -29.5285416, 16.7830086, -29.5846519, 16.8392582, -45.8108139, 45.8253784
31: -26.2358418, 19.0288868, -26.3133316, 19.0670509, -45.3028946, 45.3422165
32: -42.0976334, 8.3825626, -42.1875305, 8.4517298, -47.4000015, 47.4208679
33: -72.2371979, -5.7986298, -72.3121414, -5.6730881, -61.1312332, 61.0833969
34: -56.3911133, -5.5705414, -56.4488487, -5.5251780, -43.4633179, 43.4850578
35: -50.0421753, -0.0461674, -50.0984955, 0.0018139, -48.0961227, 48.1184349
36: -47.6191750, 4.8827810, -47.7201958, 4.9095554, -51.8130035, 51.8956680
37: -83.5477295, -17.5396843, -83.6169586, -17.4805374, -58.2698593, 58.2770195
38: -58.5006180, 3.1589432, -58.5828705, 3.1886568, -61.0746765, 61.1259308
39: -78.8437119, -11.6760168, -78.9051514, -11.6341124, -65.1555328, 65.1745605
40: -67.5763321, -18.4487686, -67.6271667, -18.3533859, -41.0972443, 41.0679016
41: -55.1152344, -6.9137602, -55.1599236, -6.8651876, -42.1824493, 42.1690216
42: -33.8962631, 6.7290010, -33.9303131, 6.8003922, -37.6225357, 37.5816193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6961665, upper bound: 44.8453589
time: 50.57 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5968077, upper bound: 44.9404289
time: 50.09 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -28.0007820, 16.9270573, -28.1562309, 17.0725517, -44.2985992, 44.2979507
1: -13.5942965, 16.9095383, -13.6877737, 17.0543480, -30.6486435, 30.5973129
2: -13.9743519, 21.4538574, -14.0969343, 21.6241570, -35.3695602, 35.3045120
3: -12.8280888, 23.2569008, -12.9103270, 23.3998394, -36.2279282, 36.1672287
4: -21.4496460, 18.2709942, -21.5784645, 18.4177284, -39.8673744, 39.8494568
5: -11.9278736, 22.6765556, -12.0188522, 22.7999344, -34.7278061, 34.6954079
6: -50.6262665, -3.6817207, -50.7735558, -3.5687571, -40.5221939, 40.5428543
7: -16.2958202, 18.2429771, -16.4090214, 18.4045582, -34.7003784, 34.6520004
8: -18.1831627, 21.1175041, -18.3354855, 21.3121071, -39.4952698, 39.4529877
9: -16.5186024, 23.1162872, -16.7041626, 23.2753925, -38.4518204, 38.4784126
10: -24.0708351, 38.3043518, -24.2805214, 38.5276489, -61.5960312, 61.5980492
11: -24.6684113, 17.5308418, -24.7705879, 17.5987930, -42.2672043, 42.3014297
12: -28.3650246, 19.9287186, -28.6070595, 20.1305256, -46.6311569, 46.6849365
13: -32.7835159, 28.6569901, -32.9330215, 28.7941036, -61.5776215, 61.5900116
14: -23.1356182, 39.0543289, -23.5012550, 39.2342453, -59.7436485, 59.9309044
15: -18.8191280, 25.7726421, -18.9533615, 25.8750916, -44.6942215, 44.7260056
16: -32.6197929, 19.7606316, -32.7350464, 19.9064827, -52.5262756, 52.4956779
17: -17.5478287, 38.3630447, -17.7679749, 38.4989891, -55.0677681, 55.1575890
18: -25.6502075, 19.5443535, -25.8309937, 19.6112289, -45.2614365, 45.3753471
19: -26.3222084, 12.4596004, -26.4706345, 12.5115395, -38.8337479, 38.9302368
20: -20.9522972, 20.3774815, -21.1411324, 20.4598618, -41.4121590, 41.5186157
21: -25.5509911, 18.8147488, -25.7588234, 18.9084873, -44.4594803, 44.5735703
22: -21.9665222, 24.4736214, -22.1443501, 24.5394592, -46.5059814, 46.6179733
23: -21.6144714, 17.4666328, -21.7052116, 17.5124931, -39.1269646, 39.1718445
24: -32.0228539, 11.8110342, -32.2012253, 11.8965044, -43.9193573, 44.0122604
25: -18.0118599, 25.3798065, -18.1242466, 25.4298325, -43.4416924, 43.5040512
26: -29.0335388, 26.8623962, -29.2316189, 26.9678154, -56.0013542, 56.0940170
27: -31.9803104, 16.4638157, -32.1417274, 16.5430946, -47.6616745, 47.7757988
28: -21.4309540, 21.6642895, -21.5442390, 21.7129822, -43.1439362, 43.2085266
29: -23.5513344, 22.1602783, -23.7210064, 22.2332726, -45.7846069, 45.8812866
30: -29.5285416, 16.7830086, -29.6711540, 16.8712139, -45.8350601, 45.9097595
31: -26.2358418, 19.0288868, -26.4325008, 19.1060410, -45.3418808, 45.4613876
32: -42.0976334, 8.3825626, -42.2622299, 8.4996405, -47.4474182, 47.4964485
33: -72.2371979, -5.7986298, -72.4623032, -5.6061487, -61.1983490, 61.2337646
34: -56.3911133, -5.5705414, -56.5451393, -5.4741459, -43.5162239, 43.5857697
35: -50.0421753, -0.0461674, -50.2202873, 0.0564013, -48.1515579, 48.2409363
36: -47.6191750, 4.8827810, -47.8208694, 4.9685650, -51.8719101, 51.9983826
37: -83.5477295, -17.5396843, -83.7299881, -17.4395962, -58.3102341, 58.3924561
38: -58.5006180, 3.1589432, -58.7415428, 3.2589159, -61.1454468, 61.2845383
39: -78.8437119, -11.6760168, -79.0694427, -11.5691299, -65.2206879, 65.3404999
40: -67.5763321, -18.4487686, -67.7210541, -18.3202705, -41.1289177, 41.1638908
41: -55.1152344, -6.9137602, -55.2216568, -6.8158684, -42.2333069, 42.2350388
42: -33.8962631, 6.7290010, -33.9440002, 6.8204136, -37.6445656, 37.5964813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 888
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 888

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6759659, upper bound: 44.8651308
time: 79.14 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5767542, upper bound: 44.8651310
time: 60.57 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -28.0478897, 16.9862556, -28.0742378, 17.0239468, -44.2874527, 44.2767067
1: -13.6329556, 16.9907875, -13.6532459, 17.0352459, -30.6682014, 30.6440334
2: -14.0315371, 21.5529575, -14.0649872, 21.6092682, -35.3954887, 35.3707962
3: -12.8692207, 23.3377934, -12.8862476, 23.3874416, -36.2566605, 36.2240410
4: -21.5154953, 18.3819199, -21.5519180, 18.4323711, -39.9478683, 39.9338379
5: -11.9703188, 22.7486534, -11.9862232, 22.7874107, -34.7577286, 34.7348785
6: -50.6611214, -3.6420722, -50.6765671, -3.6282663, -40.5258484, 40.4806786
7: -16.3456802, 18.3333015, -16.3748703, 18.3980217, -34.7437019, 34.7081718
8: -18.2449150, 21.2138901, -18.2820358, 21.2729263, -39.5178413, 39.4959259
9: -16.6257782, 23.1809521, -16.6948204, 23.2248573, -38.4966927, 38.5299721
10: -24.2178383, 38.3996887, -24.2746277, 38.4565887, -61.6620789, 61.6688080
11: -24.7388458, 17.5736217, -24.7574730, 17.5941372, -42.3329849, 42.3310928
12: -28.5256500, 20.0232143, -28.6398888, 20.0908813, -46.7431946, 46.8054543
13: -32.8475838, 28.7013397, -32.9200630, 28.7539635, -61.6015472, 61.6214027
14: -23.3055916, 39.1132050, -23.4521694, 39.1596413, -59.8292313, 59.9183311
15: -18.8808784, 25.8227692, -18.9043350, 25.8370285, -44.7179070, 44.7271042
16: -32.6805077, 19.8211689, -32.7136765, 19.8505478, -52.5310555, 52.5348434
17: -17.6554317, 38.4050674, -17.7444534, 38.4292221, -55.1039505, 55.1582336
18: -25.7240696, 19.5994167, -25.7734413, 19.6096611, -45.3337326, 45.3728561
19: -26.3715782, 12.4778690, -26.3959961, 12.4816160, -38.8531952, 38.8738632
20: -21.0217590, 20.4082794, -21.0709877, 20.4283676, -41.4501266, 41.4792671
21: -25.6288414, 18.8504181, -25.6758442, 18.8687115, -44.4975510, 44.5262604
22: -22.0261612, 24.4983253, -22.0728016, 24.5154343, -46.5415955, 46.5711288
23: -21.6528072, 17.4867001, -21.6796913, 17.4946728, -39.1474800, 39.1663895
24: -32.0671997, 11.8580570, -32.1078033, 11.8945522, -43.9617538, 43.9658585
25: -18.0611572, 25.4072990, -18.0833702, 25.4179611, -43.4791183, 43.4906693
26: -29.1247234, 26.9161739, -29.2019348, 26.9514656, -56.0761871, 56.1181107
27: -32.0398560, 16.5129185, -32.0864449, 16.5350323, -47.7094345, 47.7234650
28: -21.4811687, 21.6883430, -21.5107651, 21.6942215, -43.1753922, 43.1991081
29: -23.6256218, 22.1994152, -23.6742439, 22.2247257, -45.8503494, 45.8736572
30: -29.5894375, 16.8235893, -29.6026897, 16.8437881, -45.8845139, 45.8751831
31: -26.2884197, 19.0463104, -26.3276081, 19.0696564, -45.3580780, 45.3739166
32: -42.1708298, 8.4251547, -42.2105103, 8.4555893, -47.4963760, 47.4856987
33: -72.2764435, -5.7250462, -72.3158188, -5.6528416, -61.1907959, 61.1672134
34: -56.4205399, -5.5419903, -56.4540749, -5.5182676, -43.5120049, 43.5207977
35: -50.0737991, -0.0143509, -50.1005936, 0.0105410, -48.1640778, 48.1730309
36: -47.6735611, 4.8941698, -47.7315521, 4.9124441, -51.8764114, 51.9260788
37: -83.5907059, -17.5092678, -83.6235275, -17.4714737, -58.3472137, 58.3701248
38: -58.5602608, 3.1834955, -58.5949554, 3.1949415, -61.1232147, 61.1727676
39: -78.8844147, -11.6528339, -78.9133606, -11.6288776, -65.1987381, 65.2078934
40: -67.6025162, -18.4071827, -67.6331711, -18.3461609, -41.1252289, 41.1553688
41: -55.1405411, -6.8828945, -55.1664047, -6.8633528, -42.2364388, 42.2393379
42: -33.9468613, 6.7788181, -33.9464569, 6.8058319, -37.6993942, 37.6392593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6176812, upper bound: 44.8901932
time: 30.99 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8670846, upper bound: 45.0170935
time: 40.58 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -28.0478897, 16.9862556, -28.1611366, 17.0898991, -44.3581924, 44.3587608
1: -13.6329556, 16.9907875, -13.6897049, 17.0795383, -30.7124939, 30.6804924
2: -14.0315371, 21.5529575, -14.1004677, 21.6565361, -35.4475174, 35.4039764
3: -12.8692207, 23.3377934, -12.9127579, 23.4260559, -36.2952766, 36.2505493
4: -21.5154953, 18.3819199, -21.5846252, 18.4535904, -39.9690857, 39.9665451
5: -11.9703188, 22.7486534, -12.0214529, 22.8233585, -34.7936783, 34.7701073
6: -50.6611214, -3.6420722, -50.7838364, -3.5690541, -40.5856056, 40.5968628
7: -16.3456802, 18.3333015, -16.4115601, 18.4318314, -34.7775116, 34.7448616
8: -18.2449150, 21.2138901, -18.3407898, 21.3426857, -39.5876007, 39.5546799
9: -16.6257782, 23.1809521, -16.7342491, 23.2823677, -38.5532875, 38.5681725
10: -24.2178383, 38.3996887, -24.3256416, 38.5375595, -61.7490387, 61.7216339
11: -24.7388458, 17.5736217, -24.7918167, 17.6028652, -42.3417130, 42.3654404
12: -28.5256500, 20.0232143, -28.6585541, 20.1378765, -46.7861252, 46.8224945
13: -32.8475838, 28.7013397, -32.9435730, 28.8018494, -61.6494331, 61.6449127
14: -23.3055916, 39.1132050, -23.5508823, 39.2383423, -59.9096336, 60.0177803
15: -18.8808784, 25.8227692, -18.9549694, 25.8902664, -44.7711449, 44.7777405
16: -32.6805077, 19.8211689, -32.7518539, 19.9109783, -52.5914841, 52.5730209
17: -17.6554317, 38.4050674, -17.8002357, 38.5033340, -55.1781921, 55.2146416
18: -25.7240696, 19.5994167, -25.8436661, 19.6213226, -45.3453903, 45.4430847
19: -26.3715782, 12.4778690, -26.4850101, 12.5128040, -38.8843842, 38.9628792
20: -21.0217590, 20.4082794, -21.1611004, 20.4616165, -41.4833755, 41.5693817
21: -25.6288414, 18.8504181, -25.7831211, 18.9110336, -44.5398750, 44.6335373
22: -22.0261612, 24.4983253, -22.1566925, 24.5424118, -46.5685730, 46.6550179
23: -21.6528072, 17.4867001, -21.7159595, 17.5140495, -39.1668549, 39.2026596
24: -32.0671997, 11.8580570, -32.2073441, 11.9110222, -43.9782219, 44.0653992
25: -18.0611572, 25.4072990, -18.1372814, 25.4322643, -43.4934235, 43.5445786
26: -29.1247234, 26.9161739, -29.2541142, 26.9710503, -56.0957718, 56.1702881
27: -32.0398560, 16.5129185, -32.1504135, 16.5534916, -47.7355614, 47.7937508
28: -21.4811687, 21.6883430, -21.5586014, 21.7149010, -43.1960678, 43.2469444
29: -23.6256218, 22.1994152, -23.7422009, 22.2360249, -45.8616486, 45.9416161
30: -29.5894375, 16.8235893, -29.6891785, 16.8757534, -45.9087868, 45.9595718
31: -26.2884197, 19.0463104, -26.4467564, 19.1086235, -45.3970413, 45.4930649
32: -42.1708298, 8.4251547, -42.2851982, 8.5035000, -47.5438080, 47.5612984
33: -72.2764435, -5.7250462, -72.4659882, -5.5859470, -61.2579269, 61.3176422
34: -56.4205399, -5.5419903, -56.5503845, -5.4672604, -43.5649071, 43.6215210
35: -50.0737991, -0.0143509, -50.2223511, 0.0651226, -48.2195129, 48.2955170
36: -47.6735611, 4.8941698, -47.8322105, 4.9714508, -51.9353027, 52.0287857
37: -83.5907059, -17.5092678, -83.7365265, -17.4305592, -58.3875732, 58.4855461
38: -58.5602608, 3.1834955, -58.7536583, 3.2652140, -61.1940002, 61.3313904
39: -78.8844147, -11.6528339, -79.0776978, -11.5639372, -65.2638397, 65.3738632
40: -67.6025162, -18.4071827, -67.7270432, -18.3130550, -41.1568794, 41.2513580
41: -55.1405411, -6.8828945, -55.2281570, -6.8140354, -42.2872696, 42.3053665
42: -33.9468613, 6.7788181, -33.9601212, 6.8258657, -37.7213974, 37.6541214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 968
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.5970484, upper bound: 44.9097630
time: 46.78 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8464028, upper bound: 45.0367516
time: 32.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -28.1330185, 17.0509682, -28.1289635, 17.0284576, -44.3702316, 44.3932419
1: -13.6671629, 17.0342140, -13.6848669, 17.0385742, -30.7057381, 30.7190819
2: -14.0649414, 21.5993843, -14.0959902, 21.6127243, -35.4277420, 35.4514236
3: -12.8935890, 23.3753738, -12.9125319, 23.3933582, -36.2869492, 36.2879066
4: -21.5455170, 18.4024620, -21.5795784, 18.4412575, -39.9867744, 39.9820404
5: -12.0025063, 22.7832737, -12.0200682, 22.7926445, -34.7951508, 34.8033409
6: -50.7636642, -3.5850968, -50.6799660, -3.5763912, -40.6907425, 40.5376282
7: -16.3782558, 18.3654938, -16.4087219, 18.4019928, -34.7802505, 34.7742157
8: -18.2994881, 21.2826805, -18.3397064, 21.2787342, -39.5782242, 39.6223869
9: -16.6638680, 23.2369423, -16.7285843, 23.2293587, -38.5372124, 38.6149025
10: -24.2675571, 38.4782181, -24.3134117, 38.4668884, -61.7243652, 61.7780342
11: -24.7682285, 17.5802402, -24.7630672, 17.6054649, -42.3736954, 42.3433075
12: -28.5428925, 20.0664558, -28.6441517, 20.1255054, -46.7884293, 46.8590012
13: -32.8696747, 28.7476845, -32.9321671, 28.7717934, -61.6414680, 61.6798515
14: -23.4011803, 39.1909943, -23.5190067, 39.1641464, -59.9210052, 60.0644188
15: -18.9288597, 25.8718033, -18.9418163, 25.8441124, -44.7729721, 44.8136215
16: -32.7169800, 19.8794746, -32.7395020, 19.8593102, -52.5762901, 52.6189766
17: -17.7086906, 38.4780388, -17.7831154, 38.4338531, -55.1559486, 55.2763367
18: -25.7898216, 19.6100121, -25.7807846, 19.6180782, -45.4078979, 45.3907967
19: -26.4595184, 12.5073414, -26.4040394, 12.5114479, -38.9709663, 38.9113808
20: -21.1102180, 20.4408798, -21.0796089, 20.4565659, -41.5667839, 41.5204887
21: -25.7341270, 18.8918343, -25.6869526, 18.9083538, -44.6424789, 44.5787888
22: -22.1074200, 24.5227814, -22.0813560, 24.5416813, -46.6491013, 46.6041374
23: -21.6882229, 17.5049191, -21.6883125, 17.5049210, -39.1931458, 39.1932297
24: -32.1644897, 11.8736486, -32.1147957, 11.9111958, -44.0756836, 43.9884453
25: -18.1134758, 25.4199467, -18.0942688, 25.4338436, -43.5473175, 43.5142136
26: -29.1723976, 26.9339886, -29.2121410, 26.9716988, -56.1440964, 56.1461296
27: -32.0970383, 16.5305691, -32.0951767, 16.5527687, -47.8340683, 47.7528915
28: -21.5277042, 21.7075806, -21.5188675, 21.7130775, -43.2407837, 43.2264481
29: -23.6916637, 22.2091198, -23.6797218, 22.2367706, -45.9284363, 45.8888397
30: -29.6740208, 16.8544235, -29.6102486, 16.8664207, -45.9861832, 45.9132271
31: -26.4059277, 19.0840225, -26.3411865, 19.1068192, -45.5127487, 45.4252090
32: -42.2423859, 8.4715805, -42.2156982, 8.4934607, -47.6061897, 47.5403900
33: -72.4244537, -5.6603909, -72.3250809, -5.5860367, -61.4057617, 61.2359772
34: -56.5156174, -5.4941854, -56.4596825, -5.4672832, -43.6603432, 43.5630722
35: -50.1932487, 0.0355377, -50.1075859, 0.0685730, -48.3428650, 48.2166290
36: -47.7724495, 4.9518452, -47.7387123, 4.9720573, -52.0366058, 51.9943848
37: -83.7024155, -17.4715080, -83.6314850, -17.4336414, -58.4990692, 58.3676338
38: -58.7147408, 3.2525311, -58.6072655, 3.2648420, -61.3491516, 61.2619553
39: -79.0464172, -11.5916309, -78.9237823, -11.5635853, -65.4278946, 65.2847748
40: -67.6951141, -18.3758469, -67.6408768, -18.3199940, -41.2437401, 41.1708145
41: -55.2014351, -6.8364573, -55.1695900, -6.8218660, -42.3447990, 42.2678795
42: -33.9598312, 6.7959681, -33.9505157, 6.8199806, -37.7298470, 37.6504440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1265
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 953
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1457
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7127700, upper bound: 44.9778053
time: 57.61 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7127700, upper bound: 44.8648727
time: 63.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 123.79 seconds
IS_A1_A1_B2_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.6961665, upper bound: 44.8453589
IS_A1_A1_B2_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.5968077, upper bound: 44.9404289
IS_A1_A1_B2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.6759659, upper bound: 44.8651308
IS_A1_A1_B2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.5767542, upper bound: 44.8651310
IS_A2_A1_B2_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.6176812, upper bound: 44.8901932
IS_A2_A1_B2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.8670846, upper bound: 45.0170935
IS_A2_A1_B2_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.5970484, upper bound: 44.9097630
IS_A2_A1_B2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.8464028, upper bound: 45.0367516
IS_A2_A2_B2_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.7127700, upper bound: 44.9778053
IS_A2_A2_B2_B2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 123.79
Output dim: 14, lower bound: -44.7127700, upper bound: 44.8648727

## BFS IS instance: IS_A2_A1_B2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -28.0461235, 16.9850159, -28.0742378, 17.0239468, -44.2854385, 44.2721519
1: -13.6306992, 16.9899178, -13.6532459, 17.0352459, -30.6659451, 30.6431637
2: -14.0294638, 21.5521107, -14.0649872, 21.6092682, -35.3915787, 35.3698883
3: -12.8670864, 23.3367519, -12.8862476, 23.3874416, -36.2545280, 36.2229996
4: -21.5127983, 18.3812408, -21.5519180, 18.4323711, -39.9451675, 39.9331589
5: -11.9672737, 22.7473717, -11.9862232, 22.7874107, -34.7546844, 34.7335968
6: -50.6564102, -3.6442995, -50.6765671, -3.6282663, -40.5223923, 40.4786415
7: -16.3415794, 18.3316765, -16.3748703, 18.3980217, -34.7396011, 34.7065468
8: -18.2407341, 21.2129269, -18.2820358, 21.2729263, -39.5136604, 39.4949646
9: -16.6244583, 23.1794109, -16.6948204, 23.2248573, -38.4953499, 38.5251541
10: -24.2165489, 38.3972435, -24.2746277, 38.4565887, -61.6607971, 61.6483536
11: -24.7338657, 17.5715046, -24.7574730, 17.5941372, -42.3280029, 42.3289795
12: -28.5242500, 20.0194588, -28.6398888, 20.0908813, -46.7414856, 46.8053360
13: -32.8461533, 28.6998081, -32.9200630, 28.7539635, -61.6001167, 61.6198730
14: -23.3024502, 39.1122856, -23.4521694, 39.1596413, -59.8181686, 59.9172821
15: -18.8782768, 25.8185883, -18.9043350, 25.8370285, -44.7153053, 44.7229233
16: -32.6788406, 19.8190155, -32.7136765, 19.8505478, -52.5293884, 52.5326920
17: -17.6529083, 38.4038925, -17.7444534, 38.4292221, -55.0962219, 55.1568909
18: -25.7195415, 19.5983543, -25.7734413, 19.6096611, -45.3292007, 45.3717957
19: -26.3705139, 12.4761639, -26.3959961, 12.4816160, -38.8521309, 38.8721619
20: -21.0200958, 20.4076462, -21.0709877, 20.4283676, -41.4484634, 41.4786339
21: -25.6268368, 18.8495369, -25.6758442, 18.8687115, -44.4955482, 44.5253830
22: -22.0235386, 24.4957943, -22.0728016, 24.5154343, -46.5389709, 46.5685959
23: -21.6519566, 17.4855194, -21.6796913, 17.4946728, -39.1466293, 39.1652107
24: -32.0649567, 11.8571930, -32.1078033, 11.8945522, -43.9595108, 43.9649963
25: -18.0595207, 25.4056034, -18.0833702, 25.4179611, -43.4774818, 43.4889755
26: -29.1202488, 26.9144096, -29.2019348, 26.9514656, -56.0717163, 56.1163445
27: -32.0330162, 16.5121078, -32.0864449, 16.5350323, -47.7164841, 47.7224426
28: -21.4798508, 21.6868591, -21.5107651, 21.6942215, -43.1740723, 43.1976242
29: -23.6237450, 22.1978493, -23.6742439, 22.2247257, -45.8484726, 45.8720932
30: -29.5874920, 16.8224487, -29.6026897, 16.8437881, -45.8842621, 45.8737755
31: -26.2867584, 19.0450497, -26.3276081, 19.0696564, -45.3564148, 45.3726578
32: -42.1677170, 8.4236965, -42.2105103, 8.4555893, -47.4936676, 47.4892387
33: -72.2742615, -5.7272501, -72.3158188, -5.6528416, -61.1885681, 61.1590195
34: -56.4193268, -5.5451765, -56.4540749, -5.5182676, -43.5095634, 43.5076256
35: -50.0714874, -0.0190182, -50.1005936, 0.0105410, -48.1618958, 48.1543541
36: -47.6717987, 4.8928175, -47.7315521, 4.9124441, -51.8743668, 51.9283371
37: -83.5893936, -17.5124569, -83.6235275, -17.4714737, -58.3450928, 58.3185997
38: -58.5560265, 3.1823120, -58.5949554, 3.1949415, -61.1199646, 61.1787872
39: -78.8820724, -11.6565742, -78.9133606, -11.6288776, -65.1964188, 65.2094879
40: -67.6012268, -18.4089622, -67.6331711, -18.3461609, -41.1237793, 41.1309624
41: -55.1396713, -6.8857994, -55.1664047, -6.8633528, -42.2354622, 42.2150803
42: -33.9461670, 6.7759333, -33.9464569, 6.8058319, -37.6978378, 37.6246033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1249
type: A, layer: 1, pos: 1249
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1266
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1012
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: B, layer: 1, pos: 497
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1248
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1264
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 496
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6929218, upper bound: 44.9581555
time: 42.21 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6929218, upper bound: 44.9401724
time: 55.47 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -28.0461235, 16.9850159, -28.1611366, 17.0898991, -44.3561783, 44.3542099
1: -13.6306992, 16.9899178, -13.6897049, 17.0795383, -30.7102375, 30.6796227
2: -14.0294638, 21.5521107, -14.1004677, 21.6565361, -35.4436035, 35.4030685
3: -12.8670864, 23.3367519, -12.9127579, 23.4260559, -36.2931442, 36.2495117
4: -21.5127983, 18.3812408, -21.5846252, 18.4535904, -39.9663887, 39.9658661
5: -11.9672737, 22.7473717, -12.0214529, 22.8233585, -34.7906342, 34.7688255
6: -50.6564102, -3.6442995, -50.7838364, -3.5690541, -40.5821609, 40.5948257
7: -16.3415794, 18.3316765, -16.4115601, 18.4318314, -34.7734108, 34.7432365
8: -18.2407341, 21.2129269, -18.3407898, 21.3426857, -39.5834198, 39.5537186
9: -16.6244583, 23.1794109, -16.7342491, 23.2823677, -38.5519447, 38.5633621
10: -24.2165489, 38.3972435, -24.3256416, 38.5375595, -61.7477570, 61.7011719
11: -24.7338657, 17.5715046, -24.7918167, 17.6028652, -42.3367310, 42.3633194
12: -28.5242500, 20.0194588, -28.6585541, 20.1378765, -46.7844162, 46.8223801
13: -32.8461533, 28.6998081, -32.9435730, 28.8018494, -61.6480026, 61.6433792
14: -23.3024502, 39.1122856, -23.5508823, 39.2383423, -59.8985519, 60.0167313
15: -18.8782768, 25.8185883, -18.9549694, 25.8902664, -44.7685432, 44.7735596
16: -32.6788406, 19.8190155, -32.7518539, 19.9109783, -52.5898209, 52.5708694
17: -17.6529083, 38.4038925, -17.8002357, 38.5033340, -55.1704483, 55.2133026
18: -25.7195415, 19.5983543, -25.8436661, 19.6213226, -45.3408661, 45.4420204
19: -26.3705139, 12.4761639, -26.4850101, 12.5128040, -38.8833160, 38.9611740
20: -21.0200958, 20.4076462, -21.1611004, 20.4616165, -41.4817123, 41.5687485
21: -25.6268368, 18.8495369, -25.7831211, 18.9110336, -44.5378723, 44.6326599
22: -22.0235386, 24.4957943, -22.1566925, 24.5424118, -46.5659485, 46.6524887
23: -21.6519566, 17.4855194, -21.7159595, 17.5140495, -39.1660080, 39.2014771
24: -32.0649567, 11.8571930, -32.2073441, 11.9110222, -43.9759789, 44.0645370
25: -18.0595207, 25.4056034, -18.1372814, 25.4322643, -43.4917831, 43.5428848
26: -29.1202488, 26.9144096, -29.2541142, 26.9710503, -56.0913010, 56.1685257
27: -32.0330162, 16.5121078, -32.1504135, 16.5534916, -47.7426033, 47.7927284
28: -21.4798508, 21.6868591, -21.5586014, 21.7149010, -43.1947517, 43.2454605
29: -23.6237450, 22.1978493, -23.7422009, 22.2360249, -45.8597717, 45.9400482
30: -29.5874920, 16.8224487, -29.6891785, 16.8757534, -45.9085312, 45.9581566
31: -26.2867584, 19.0450497, -26.4467564, 19.1086235, -45.3953819, 45.4918060
32: -42.1677170, 8.4236965, -42.2851982, 8.5035000, -47.5410995, 47.5648460
33: -72.2742615, -5.7272501, -72.4659882, -5.5859470, -61.2556992, 61.3094254
34: -56.4193268, -5.5451765, -56.5503845, -5.4672604, -43.5624580, 43.6083374
35: -50.0714874, -0.0190182, -50.2223511, 0.0651226, -48.2173157, 48.2768288
36: -47.6717987, 4.8928175, -47.8322105, 4.9714508, -51.9332733, 52.0310287
37: -83.5893936, -17.5124569, -83.7365265, -17.4305592, -58.3854523, 58.4340324
38: -58.5560265, 3.1823120, -58.7536583, 3.2652140, -61.1907501, 61.3374100
39: -78.8820724, -11.6565742, -79.0776978, -11.5639372, -65.2615356, 65.3754654
40: -67.6012268, -18.4089622, -67.7270432, -18.3130550, -41.1554298, 41.2269478
41: -55.1396713, -6.8857994, -55.2281570, -6.8140354, -42.2863007, 42.2811050
42: -33.9461670, 6.7759333, -33.9601212, 6.8258657, -37.7198410, 37.6394577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 936
type: A, layer: 1, pos: 936
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 202
type: A, layer: 1, pos: 202
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 872
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 872
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1265
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 953
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 1266
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 968
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1248
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2016
type: B, layer: 1, pos: 2016
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1264
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1457
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 496
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1442
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1304

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 888

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6727998, upper bound: 44.9778055
time: 44.56 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.6727998, upper bound: 44.9598956
time: 42.63 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 89.66 seconds
IS_A2_A1_B2_B2_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 9, time: 89.66
Output dim: 14, lower bound: -44.6929218, upper bound: 44.9581555
IS_A2_A1_B2_B2_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 89.66
Output dim: 14, lower bound: -44.6929218, upper bound: 44.9401724
IS_A2_A1_B2_B2_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 9, time: 89.66
Output dim: 14, lower bound: -44.6727998, upper bound: 44.9778055
IS_A2_A1_B2_B2_B2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 89.66
Output dim: 14, lower bound: -44.6727998, upper bound: 44.9598956

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 64.18 + 4081.60 = 4145.79 seconds
