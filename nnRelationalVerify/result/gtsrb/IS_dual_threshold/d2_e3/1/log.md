## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 20.1836426535


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498)
1: (-11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070)
2: (-9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425)
3: (-9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4056396, 38.4056396)
4: (-16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9559555, 41.9559555)
5: (-7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1119690, 36.1119690)
6: (-38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812)
7: (-11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.6014786, 38.6014748)
8: (-21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7545319, 50.7545319)
9: (-13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186)
10: (-22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662)
11: (-23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790)
12: (-44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2175751, 45.2175751)
13: (-37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5576782, 59.5576782)
14: (-64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507)
15: (-21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186)
16: (-23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021)
17: (-58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0626373, 56.0626373)
18: (-35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650)
19: (-26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775)
20: (-21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115)
21: (-27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035)
22: (-32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925)
23: (-24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843)
24: (-30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705)
25: (-28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095)
26: (-41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693)
27: (-26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029)
28: (-25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052)
29: (-27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4229507, 38.4229507)
30: (-26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750)
31: (-35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546)
32: (-35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8387909, 45.8387985)
33: (-63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3612823, 55.3612900)
34: (-57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5909424, 47.5909424)
35: (-56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8676910, 44.8676949)
36: (-53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4313049, 49.4312973)
37: (-78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8089600, 60.8089523)
38: (-63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6764984, 59.6764984)
39: (-72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0168457, 58.0168381)
40: (-51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460)
41: (-40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611)
42: (-26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.92 + 57.15 = 60.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -20.2038465, upper bound: 20.2038465

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1948068, upper bound: 20.1391674
time: 49.40 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1948068, upper bound: 20.2009809
time: 70.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 119.88 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 119.88
Output dim: 5, lower bound: -20.1948068, upper bound: 20.1391674
IS_B2, status: Status.UNKNOWN, split count: 1, time: 119.88
Output dim: 5, lower bound: -20.1948068, upper bound: 20.2009809

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -37.6791458, 17.5932503, -37.6382751, 17.5744839, -55.2536316, 55.2315254
1: -11.9455767, 22.4777203, -11.9135551, 22.4706497, -34.4162254, 34.3912735
2: -9.7299509, 25.2847767, -9.6774540, 25.2769547, -35.0069046, 34.9622307
3: -9.5997715, 28.9633102, -9.5335999, 28.9514198, -38.3231049, 38.2677917
4: -16.6345634, 25.3676548, -16.5670700, 25.3575821, -41.8758774, 41.8182411
5: -7.4402113, 29.0291824, -7.3777080, 29.0151806, -36.0298767, 35.9812698
6: -38.2508087, 12.0123425, -38.2353668, 11.9772367, -50.2280464, 50.2477112
7: -11.1045895, 28.6647797, -11.0621729, 28.6554737, -38.5440750, 38.5134506
8: -21.2513027, 29.8665390, -21.1911068, 29.8577232, -50.6827469, 50.6292572
9: -13.7477055, 28.3291779, -13.7240763, 28.3075924, -42.0552979, 42.0532532
10: -22.1198311, 31.9969864, -22.0972672, 31.9493675, -54.0691986, 54.0942535
11: -23.7278671, 14.6821079, -23.7029495, 14.6085491, -38.3364182, 38.3850555
12: -44.2626953, 4.3724718, -44.2497101, 4.2523060, -44.9751968, 45.0849075
13: -37.4810257, 22.3149681, -37.4660835, 22.2577209, -59.4336472, 59.4742126
14: -64.9068756, 2.6547756, -64.8790207, 2.5434523, -67.4503250, 67.5337982
15: -21.8599243, 20.3574944, -21.7882576, 20.3370819, -42.1970062, 42.1457520
16: -23.4618492, 21.7365417, -23.4331970, 21.6997566, -45.1616058, 45.1697388
17: -58.4137878, -1.2254486, -58.3988037, -1.3222122, -55.8612671, 55.9444237
18: -35.8651047, 14.6384020, -35.8489456, 14.6183376, -50.4834442, 50.4873466
19: -26.4494438, 9.4757166, -26.4344444, 9.4368944, -35.8863373, 35.9101601
20: -21.5593052, 15.8790989, -21.5376530, 15.8358345, -37.3951416, 37.4167519
21: -27.2987766, 12.9558678, -27.2797585, 12.9025078, -40.2012863, 40.2356262
22: -32.1217651, 10.6265631, -32.1002808, 10.6065350, -42.7283020, 42.7268448
23: -24.6032009, 14.0291805, -24.5893326, 13.9974585, -38.6006584, 38.6185150
24: -30.7610264, 13.7349377, -30.7401772, 13.7240047, -44.4850311, 44.4751129
25: -28.9118767, 12.9175291, -28.8940201, 12.8859119, -41.7977905, 41.8115501
26: -41.0479965, 17.0360298, -41.0249710, 16.9784622, -58.0264587, 58.0610008
27: -26.1095467, 18.1909599, -26.0718613, 18.1813164, -44.2908630, 44.2628212
28: -25.0892601, 17.3111782, -25.0722923, 17.2797661, -42.3690262, 42.3834686
29: -27.6261921, 10.9297428, -27.6132011, 10.8896141, -38.3348465, 38.3614349
30: -26.8647556, 18.3243523, -26.8439808, 18.2948647, -45.1596222, 45.1683350
31: -35.4245644, 12.1122341, -35.4059448, 12.0682755, -47.4928398, 47.5181808
32: -35.2419891, 10.9959841, -35.2254143, 10.9485016, -45.7314301, 45.7640076
33: -63.6967239, -3.7572374, -63.6518211, -3.7802582, -55.2706451, 55.2449417
34: -57.8287811, -6.3685789, -57.7887344, -6.3891497, -47.5096283, 47.4823227
35: -56.0933113, -4.3498335, -56.0786934, -4.3652163, -44.8096161, 44.8005142
36: -53.4984932, 0.8654499, -53.4860344, 0.8299932, -49.3503418, 49.3741837
37: -78.2874146, -14.2721834, -78.2643738, -14.3095951, -60.7276306, 60.7319260
38: -63.8347626, 0.3921337, -63.8174400, 0.3482962, -59.5861053, 59.6074066
39: -72.1506348, -8.1742992, -72.1255112, -8.2018290, -57.9471283, 57.9451752
40: -51.3757629, -6.2104297, -51.3515511, -6.2260165, -45.1497459, 45.1411209
41: -40.0693207, 12.2546978, -40.0514183, 12.2311277, -52.3004494, 52.3061142
42: -26.1762428, 11.9601316, -26.1608162, 11.9289856, -38.1052284, 38.1209488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1433

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1915915, upper bound: 20.0851744
time: 53.79 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1915915, upper bound: 20.1358450
time: 56.55 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -37.7056999, 17.6068840, -37.7496605, 17.6441803, -55.3498802, 55.3565445
1: -11.9637852, 22.4827576, -12.0046806, 22.5328045, -34.4965897, 34.4874382
2: -9.7673388, 25.2899170, -9.7977486, 25.3766251, -35.1439629, 35.0876656
3: -9.6488018, 28.9703140, -9.6718359, 29.0955124, -38.5161972, 38.4088974
4: -16.6832924, 25.3741493, -16.7155685, 25.5171165, -42.0849915, 41.9693069
5: -7.4857950, 29.0382023, -7.5137315, 29.1633663, -36.2251053, 36.1189003
6: -38.2607498, 12.0309334, -38.3427429, 12.0630569, -50.3238068, 50.3736763
7: -11.1353788, 28.6704636, -11.1983471, 28.7107868, -38.6345062, 38.6451721
8: -21.2963104, 29.8718033, -21.3381748, 29.9821987, -50.8562927, 50.7731781
9: -13.7641916, 28.3395958, -13.7917690, 28.3723221, -42.1365128, 42.1313629
10: -22.1355019, 32.0246124, -22.2193794, 32.0581245, -54.1936264, 54.2439919
11: -23.7449608, 14.7386093, -23.9940300, 14.7619839, -38.5069427, 38.7326393
12: -44.2714462, 4.4635162, -44.5388374, 4.5109806, -45.2244949, 45.4673538
13: -37.4901237, 22.3561554, -37.5825806, 22.4113350, -59.5747147, 59.6550827
14: -64.9233780, 2.7419424, -65.1690979, 2.7637615, -67.6871414, 67.9110413
15: -21.9130650, 20.3714447, -21.9579887, 20.5450916, -42.4581566, 42.3294334
16: -23.4802132, 21.7523632, -23.5866184, 21.7653217, -45.2455368, 45.3389816
17: -58.4234238, -1.1523800, -58.6971550, -1.1192093, -56.0451584, 56.3461304
18: -35.8714218, 14.6525974, -35.9291801, 14.6742992, -50.5457230, 50.5817795
19: -26.4594536, 9.5041466, -26.5694122, 9.5213757, -35.9808273, 36.0735588
20: -21.5746651, 15.9100914, -21.6659145, 15.9235601, -37.4982262, 37.5760040
21: -27.3115845, 12.9957142, -27.4702721, 13.0176783, -40.3292618, 40.4659882
22: -32.1348877, 10.6351318, -32.1817207, 10.6841841, -42.8190727, 42.8168526
23: -24.6123924, 14.0521317, -24.7047901, 14.0704880, -38.6828804, 38.7569199
24: -30.7700577, 13.7426224, -30.8073368, 13.7624130, -44.5324707, 44.5499573
25: -28.9220619, 12.9403458, -28.9824104, 12.9683523, -41.8904152, 41.9227562
26: -41.0631638, 17.0747356, -41.1793060, 17.0977268, -58.1608887, 58.2540436
27: -26.1304302, 18.1973686, -26.1766624, 18.2431259, -44.3735580, 44.3740311
28: -25.1009960, 17.3330765, -25.1709099, 17.3478832, -42.4488792, 42.5039864
29: -27.6321507, 10.9599876, -27.7262878, 10.9846144, -38.4327240, 38.5078278
30: -26.8775330, 18.3341484, -26.9435787, 18.3527737, -45.2303085, 45.2777252
31: -35.4368515, 12.1447277, -35.5505180, 12.1589489, -47.5958023, 47.6952438
32: -35.2521515, 11.0289192, -35.3582649, 11.0549717, -45.8463974, 45.9405212
33: -63.7271957, -3.7415228, -63.7570267, -3.6513286, -55.4746552, 55.3612137
34: -57.8583527, -6.3554354, -57.8891106, -6.2396250, -47.7402878, 47.5874786
35: -56.1021805, -4.3399420, -56.1235847, -4.2851858, -44.9583511, 44.8513908
36: -53.5063858, 0.8912067, -53.5936127, 0.9340706, -49.4496460, 49.5088425
37: -78.3006973, -14.2459478, -78.3905106, -14.2087650, -60.8475342, 60.8618393
38: -63.8450928, 0.4238009, -63.9376335, 0.4755068, -59.7040710, 59.7624969
39: -72.1661072, -8.1604004, -72.2366486, -8.1005487, -58.0568848, 58.0854797
40: -51.3855286, -6.2022729, -51.4378090, -6.1263266, -45.2592010, 45.2355347
41: -40.0810280, 12.2688379, -40.1322098, 12.2943897, -52.3754196, 52.4010468
42: -26.1874390, 11.9762325, -26.2531242, 12.0022173, -38.1896553, 38.2293549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1433

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1915915, upper bound: 20.0851744
time: 51.46 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1975808, upper bound: 20.1975798
time: 50.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 104.49 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 104.49
Output dim: 5, lower bound: -20.1915915, upper bound: 20.0851744
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 104.49
Output dim: 5, lower bound: -20.1915915, upper bound: 20.1358450
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 104.49
Output dim: 5, lower bound: -20.1915915, upper bound: 20.0851744
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 104.49
Output dim: 5, lower bound: -20.1975808, upper bound: 20.1975798

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -37.6758041, 17.5922031, -37.5989189, 17.5620937, -55.2378998, 55.1911240
1: -11.9422626, 22.4771461, -11.8745222, 22.4637051, -34.4059677, 34.3516693
2: -9.7256002, 25.2842560, -9.6261940, 25.2707977, -34.9963989, 34.9104500
3: -9.5948343, 28.9624863, -9.4760361, 28.9416599, -38.3086090, 38.2094879
4: -16.6300602, 25.3669472, -16.5154419, 25.3492374, -41.8630676, 41.7665024
5: -7.4352770, 29.0284615, -7.3212476, 29.0064621, -36.0162659, 35.9225540
6: -38.2496681, 12.0109425, -38.2220116, 11.9612103, -50.2108765, 50.2329559
7: -11.1001797, 28.6641541, -11.0123682, 28.6480026, -38.5317993, 38.4600983
8: -21.2465363, 29.8657608, -21.1349449, 29.8487186, -50.6692047, 50.5696411
9: -13.7462940, 28.3255863, -13.7073612, 28.2657280, -42.0120239, 42.0329475
10: -22.1178570, 31.9894104, -22.0736523, 31.8604679, -53.9783249, 54.0630646
11: -23.7266521, 14.6784878, -23.6886024, 14.5673428, -38.2939949, 38.3670883
12: -44.2616386, 4.3639803, -44.2374840, 4.1527662, -44.8740234, 45.0641937
13: -37.4798279, 22.3133316, -37.4524574, 22.2384987, -59.4079361, 59.4571075
14: -64.9041595, 2.6466322, -64.8475037, 2.4479618, -67.3521194, 67.4941330
15: -21.8571548, 20.3561535, -21.7557564, 20.3212051, -42.1783600, 42.1119080
16: -23.4598465, 21.7335606, -23.4094410, 21.6651802, -45.1250267, 45.1430016
17: -58.4127769, -1.2295723, -58.3871994, -1.3708696, -55.8126068, 55.9286842
18: -35.8640366, 14.6355686, -35.8360634, 14.5843544, -50.4483910, 50.4716339
19: -26.4483547, 9.4729958, -26.4213905, 9.4049911, -35.8533478, 35.8943863
20: -21.5577660, 15.8758841, -21.5193825, 15.7978592, -37.3556252, 37.3952675
21: -27.2975006, 12.9518414, -27.2646561, 12.8552914, -40.1527939, 40.2164993
22: -32.1208115, 10.6244354, -32.0893784, 10.5813560, -42.7021675, 42.7138138
23: -24.6021328, 14.0268154, -24.5767708, 13.9697037, -38.5718384, 38.6035843
24: -30.7599640, 13.7343693, -30.7277985, 13.7169561, -44.4769211, 44.4621658
25: -28.9109077, 12.9150915, -28.8826866, 12.8575745, -41.7684822, 41.7977791
26: -41.0465927, 17.0297356, -41.0085449, 16.9045677, -57.9511604, 58.0382805
27: -26.1072254, 18.1903248, -26.0444717, 18.1738853, -44.2811127, 44.2347946
28: -25.0879688, 17.3091679, -25.0571518, 17.2565022, -42.3444710, 42.3663177
29: -27.6254272, 10.9272270, -27.6044350, 10.8601875, -38.3043594, 38.3499527
30: -26.8636723, 18.3219414, -26.8312435, 18.2666969, -45.1303711, 45.1531830
31: -35.4229813, 12.1088581, -35.3872070, 12.0281811, -47.4511642, 47.4960632
32: -35.2408485, 10.9933300, -35.2123184, 10.9173946, -45.6960297, 45.7482376
33: -63.6931000, -3.7589006, -63.6089973, -3.7995172, -55.2477951, 55.2006149
34: -57.8263664, -6.3699188, -57.7606049, -6.4049263, -47.4917984, 47.4558716
35: -56.0911026, -4.3509560, -56.0528374, -4.3786774, -44.7947083, 44.7727737
36: -53.4975662, 0.8647079, -53.4749641, 0.8211174, -49.3402023, 49.3631439
37: -78.2861633, -14.2740660, -78.2494507, -14.3313522, -60.7054291, 60.7139511
38: -63.8328476, 0.3911057, -63.7954254, 0.3370485, -59.5728149, 59.5848846
39: -72.1490021, -8.1754360, -72.1064301, -8.2155256, -57.9271774, 57.9271927
40: -51.3742828, -6.2110777, -51.3340683, -6.2337394, -45.1405449, 45.1229897
41: -40.0676117, 12.2535381, -40.0315475, 12.2172813, -52.2848930, 52.2850876
42: -26.1753616, 11.9581575, -26.1506424, 11.9056902, -38.0810509, 38.1087990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=261, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1433

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1408032, upper bound: 20.0851744
time: 48.27 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1408032, upper bound: 20.0851744
time: 75.96 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -37.6647835, 17.5905228, -37.6750107, 17.6151257, -55.2799072, 55.2655334
1: -11.9381866, 22.4763680, -11.9441967, 22.6080017, -34.5461884, 34.4205627
2: -9.7215853, 25.2830620, -9.6993847, 25.4314804, -35.1530647, 34.9824448
3: -9.5896721, 28.9600868, -9.5444984, 29.1303082, -38.4992981, 38.2745895
4: -16.6263733, 25.3648129, -16.6033516, 25.5383396, -42.0492249, 41.8525314
5: -7.4304781, 29.0266514, -7.4028144, 29.1602077, -36.1691322, 36.0013771
6: -38.2467346, 12.0029631, -38.3059692, 11.9983959, -50.2451324, 50.3089333
7: -11.0936871, 28.6625290, -11.1146984, 28.7617245, -38.6526794, 38.5619965
8: -21.2421074, 29.8641701, -21.2188911, 30.0439796, -50.8750610, 50.6461563
9: -13.7427683, 28.3223820, -13.8783083, 28.3473034, -42.0900726, 42.2006912
10: -22.1149693, 31.9826717, -22.4472141, 31.9725647, -54.0863266, 54.4298859
11: -23.7243557, 14.6578112, -23.9418316, 14.5907631, -38.3151169, 38.5996437
12: -44.2593765, 4.3581266, -44.5614738, 4.2830524, -44.9990387, 45.3894730
13: -37.4763107, 22.3079262, -37.4907913, 22.3289948, -59.4757614, 59.5122910
14: -64.8997955, 2.6408033, -65.2236633, 2.5560112, -67.4558105, 67.8644638
15: -21.8352032, 20.3542805, -21.7943459, 20.4278011, -42.2630043, 42.1486282
16: -23.4552956, 21.7180576, -23.6021652, 21.6974926, -45.1527863, 45.3202209
17: -58.4115868, -1.2348785, -58.6363564, -1.2900829, -55.8811646, 56.1925125
18: -35.8624115, 14.6295757, -36.0087433, 14.6283054, -50.4907150, 50.6383209
19: -26.4467201, 9.4695454, -26.5829277, 9.4457111, -35.8924332, 36.0524750
20: -21.5557461, 15.8723421, -21.6838379, 15.8408957, -37.3966408, 37.5561790
21: -27.2950706, 12.9483681, -27.4925137, 12.9198990, -40.2149696, 40.4408798
22: -32.1183167, 10.6131802, -32.1731262, 10.6316996, -42.7500153, 42.7863083
23: -24.6004639, 14.0214548, -24.6974163, 14.0095100, -38.6099739, 38.7188721
24: -30.7559586, 13.7315617, -30.7814617, 13.7326517, -44.4886093, 44.5130234
25: -28.9085827, 12.9117470, -28.9761238, 12.9116631, -41.8202438, 41.8878708
26: -41.0433502, 17.0208626, -41.2303772, 16.9992981, -58.0426483, 58.2512398
27: -26.1012573, 18.1888809, -26.1203671, 18.2365627, -44.3378220, 44.3092499
28: -25.0863533, 17.3043804, -25.1329784, 17.2895679, -42.3759232, 42.4373589
29: -27.6222477, 10.9192123, -27.7161636, 10.9022465, -38.3445663, 38.4574127
30: -26.8611145, 18.3098812, -26.9346581, 18.2974510, -45.1585655, 45.2445374
31: -35.4211349, 12.1038513, -35.5822296, 12.0716705, -47.4928055, 47.6860809
32: -35.2379761, 10.9912968, -35.3458748, 10.9750500, -45.7400055, 45.8922729
33: -63.6893387, -3.7617745, -63.6728134, -3.6114049, -55.4783783, 55.2611389
34: -57.8237991, -6.3723249, -57.8152237, -6.2588978, -47.6778259, 47.5059204
35: -56.0859146, -4.3519678, -56.0867882, -4.2232494, -45.0006104, 44.8143616
36: -53.4929695, 0.8640118, -53.5069580, 0.8912048, -49.4066315, 49.3948822
37: -78.2817535, -14.2801371, -78.3204651, -14.2841091, -60.7691345, 60.7964706
38: -63.8283615, 0.3892579, -63.8562012, 0.4250445, -59.6577606, 59.6413574
39: -72.1439743, -8.1769638, -72.1723099, -8.1215353, -58.0175323, 58.0165176
40: -51.3709869, -6.2131610, -51.4258842, -6.1652985, -45.2056885, 45.2127228
41: -40.0609894, 12.2522278, -40.0911179, 12.2773972, -52.3383865, 52.3433456
42: -26.1730461, 11.9444914, -26.2284737, 11.9522057, -38.1252518, 38.1729660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=261, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1400

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 733

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1480517, upper bound: 20.1338354
time: 46.36 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1895822, upper bound: 20.1338354
time: 57.28 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -37.7023163, 17.6058559, -37.7103615, 17.6318359, -55.3341522, 55.3162155
1: -11.9604530, 22.4821968, -11.9657211, 22.5259247, -34.4863777, 34.4479179
2: -9.7629528, 25.2893791, -9.7466259, 25.3705330, -35.1334839, 35.0360031
3: -9.6438646, 28.9694901, -9.6143894, 29.0858650, -38.5018082, 38.3506927
4: -16.6787930, 25.3734474, -16.6643181, 25.5088539, -42.0722351, 41.9180145
5: -7.4808602, 29.0374756, -7.4572959, 29.1546898, -36.2115746, 36.0602875
6: -38.2596130, 12.0295630, -38.3294716, 12.0469923, -50.3066063, 50.3590355
7: -11.1309795, 28.6698303, -11.1485672, 28.7033806, -38.6223526, 38.5919037
8: -21.2915154, 29.8710289, -21.2822361, 29.9732609, -50.8428040, 50.7136688
9: -13.7627869, 28.3360119, -13.7752028, 28.3305779, -42.0933647, 42.1112137
10: -22.1334782, 32.0170441, -22.1959763, 31.9696083, -54.1030884, 54.2130203
11: -23.7437420, 14.7349653, -23.9798851, 14.7207966, -38.4645386, 38.7148514
12: -44.2704201, 4.4550095, -44.5266953, 4.4116354, -45.1234131, 45.4467545
13: -37.4889641, 22.3545532, -37.5690613, 22.3918762, -59.5489349, 59.6379852
14: -64.9207001, 2.7338524, -65.1376801, 2.6683207, -67.5890198, 67.8715363
15: -21.9102726, 20.3700962, -21.9252663, 20.5295944, -42.4398651, 42.2953644
16: -23.4781857, 21.7494144, -23.5628967, 21.7303352, -45.2085190, 45.3123093
17: -58.4224167, -1.1565180, -58.6855392, -1.1677999, -55.9963684, 56.3304138
18: -35.8703613, 14.6497221, -35.9162064, 14.6403675, -50.5107269, 50.5659294
19: -26.4583626, 9.5014610, -26.5564671, 9.4895496, -35.9479141, 36.0579300
20: -21.5731049, 15.9068460, -21.6477699, 15.8856583, -37.4587631, 37.5546150
21: -27.3102913, 12.9916611, -27.4552746, 12.9705515, -40.2808418, 40.4469376
22: -32.1339569, 10.6330137, -32.1707878, 10.6590414, -42.7929993, 42.8038025
23: -24.6113167, 14.0497780, -24.6922855, 14.0428152, -38.6541328, 38.7420654
24: -30.7689915, 13.7420053, -30.7950249, 13.7554350, -44.5244255, 44.5370293
25: -28.9211006, 12.9379005, -28.9712372, 12.9401140, -41.8612137, 41.9091377
26: -41.0617905, 17.0684624, -41.1629562, 17.0237999, -58.0855904, 58.2314186
27: -26.1280804, 18.1967239, -26.1496220, 18.2357693, -44.3638496, 44.3463440
28: -25.0997066, 17.3310528, -25.1558552, 17.3247604, -42.4244690, 42.4869080
29: -27.6313896, 10.9574852, -27.7175636, 10.9551792, -38.4024200, 38.4963531
30: -26.8764763, 18.3317509, -26.9309635, 18.3245258, -45.2010040, 45.2627144
31: -35.4352608, 12.1413021, -35.5318146, 12.1188564, -47.5541153, 47.6731186
32: -35.2510185, 11.0262737, -35.3452721, 11.0239277, -45.8109894, 45.9248352
33: -63.7235641, -3.7431192, -63.7141953, -3.6704903, -55.4519043, 55.3168259
34: -57.8559265, -6.3567829, -57.8608856, -6.2553644, -47.7224655, 47.5609741
35: -56.0999527, -4.3410845, -56.0976105, -4.2985907, -44.9435272, 44.8236542
36: -53.5053940, 0.8904819, -53.5825005, 0.9249563, -49.4392166, 49.4977951
37: -78.2994156, -14.2478294, -78.3756561, -14.2304573, -60.8253784, 60.8439331
38: -63.8431778, 0.4229159, -63.9154739, 0.4641418, -59.6906128, 59.7399063
39: -72.1644592, -8.1615620, -72.2177048, -8.1143332, -58.0371246, 58.0678024
40: -51.3840485, -6.2029490, -51.4203644, -6.1341543, -45.2498932, 45.2174149
41: -40.0793228, 12.2676048, -40.1123810, 12.2805481, -52.3598709, 52.3799858
42: -26.1865997, 11.9742479, -26.2430267, 11.9789267, -38.1655273, 38.2172737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=261, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1433

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1408027
time: 52.45 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1358454, upper bound: 20.0940351
time: 54.01 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -37.6912918, 17.6041756, -37.7869415, 17.6849670, -55.3762589, 55.3911171
1: -11.9563951, 22.4814034, -12.0358181, 22.6702728, -34.6266670, 34.5172195
2: -9.7589645, 25.2881851, -9.8205633, 25.5312386, -35.2902031, 35.1087494
3: -9.6387138, 28.9670830, -9.6830425, 29.2746544, -38.6924896, 38.4162674
4: -16.6750908, 25.3713150, -16.7527771, 25.6980553, -42.2584534, 42.0043716
5: -7.4760652, 29.0356712, -7.5392194, 29.3086128, -36.3646011, 36.1393356
6: -38.2566795, 12.0215445, -38.4148369, 12.0841856, -50.3408661, 50.4363823
7: -11.1244774, 28.6682014, -11.2508888, 28.8171768, -38.7432785, 38.6935310
8: -21.2871132, 29.8694534, -21.3670597, 30.1687698, -51.0488815, 50.7906265
9: -13.7592850, 28.3328323, -13.9462061, 28.4120865, -42.1713715, 42.2790375
10: -22.1306152, 32.0103188, -22.5696926, 32.0835609, -54.2060013, 54.5800095
11: -23.7414627, 14.7142944, -24.2344532, 14.7444267, -38.4858894, 38.9487457
12: -44.2681503, 4.4491386, -44.8508148, 4.5418711, -45.2483215, 45.7721252
13: -37.4854012, 22.3491230, -37.6071968, 22.4815483, -59.6159592, 59.6928864
14: -64.9163666, 2.7280293, -65.5135117, 2.7758503, -67.6922150, 68.2415390
15: -21.8882828, 20.3682022, -21.9636936, 20.6388512, -42.5271339, 42.3318939
16: -23.4736748, 21.7338886, -23.7563953, 21.7632580, -45.2369308, 45.4902840
17: -58.4212532, -1.1617708, -58.9349403, -1.0872889, -56.0649719, 56.5942612
18: -35.8687363, 14.6437645, -36.0888824, 14.6846046, -50.5533409, 50.7326469
19: -26.4567108, 9.4980106, -26.7184391, 9.5309238, -35.9876328, 36.2164497
20: -21.5711040, 15.9032993, -21.8123398, 15.9288893, -37.4999924, 37.7156372
21: -27.3078785, 12.9882145, -27.6834335, 13.0356693, -40.3435478, 40.6716461
22: -32.1314316, 10.6217365, -32.2549133, 10.7099876, -42.8414192, 42.8766479
23: -24.6096439, 14.0444231, -24.8135300, 14.0831165, -38.6927605, 38.8579521
24: -30.7649670, 13.7391748, -30.8490391, 13.7711840, -44.5361519, 44.5882149
25: -28.9187813, 12.9345531, -29.0649223, 12.9943647, -41.9131470, 41.9994736
26: -41.0585594, 17.0595970, -41.3847961, 17.1182690, -58.1768265, 58.4443932
27: -26.1221352, 18.1952724, -26.2252922, 18.2984486, -44.4205856, 44.4205627
28: -25.0980911, 17.3263092, -25.2318401, 17.3582363, -42.4563293, 42.5581512
29: -27.6282425, 10.9494781, -27.8300304, 10.9975071, -38.4428635, 38.6045151
30: -26.8739071, 18.3196449, -27.0348873, 18.3548126, -45.2287216, 45.3545303
31: -35.4334183, 12.1362734, -35.7277298, 12.1629467, -47.5963669, 47.8640022
32: -35.2481461, 11.0242338, -35.4790726, 11.0814457, -45.8549805, 46.0691299
33: -63.7197723, -3.7460423, -63.7777939, -3.4826007, -55.6822052, 55.3771515
34: -57.8533745, -6.3591957, -57.9153976, -6.1095772, -47.9083633, 47.6111298
35: -56.0947876, -4.3420982, -56.1316032, -4.1429443, -45.1496201, 44.8651390
36: -53.5008125, 0.8898582, -53.6146507, 0.9949770, -49.5058746, 49.5297318
37: -78.2950439, -14.2538853, -78.4475555, -14.1827555, -60.8890076, 60.9284592
38: -63.8386993, 0.4209518, -63.9752769, 0.5520248, -59.7756653, 59.7959824
39: -72.1593475, -8.1630669, -72.2843094, -8.0216751, -58.1262054, 58.1590118
40: -51.3807220, -6.2050233, -51.5117798, -6.0661950, -45.3145256, 45.3067551
41: -40.0726738, 12.2663774, -40.1727638, 12.3402367, -52.4129105, 52.4391403
42: -26.1842823, 11.9605913, -26.3220310, 12.0252666, -38.2095490, 38.2826233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=261, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1433

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1915911
time: 57.12 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1431459
time: 53.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 113.14 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1408032, upper bound: 20.0851744
IS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1408032, upper bound: 20.0851744
IS_B1_B2_B1, status: Status.VERIFIED, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1480517, upper bound: 20.1338354
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1895822, upper bound: 20.1338354
IS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1408027
IS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1358454, upper bound: 20.0940351
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1915911
IS_B2_B2_A2, status: Status.VERIFIED, split count: 3, time: 113.14
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1431459

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -37.6585617, 17.5842590, -37.8508530, 17.6225319, -55.2810936, 55.4351120
1: -11.9353647, 22.4735985, -12.0753365, 22.6118031, -34.5471687, 34.5489349
2: -9.7182751, 25.2803802, -9.8119640, 25.4362755, -35.1545486, 35.0923462
3: -9.5866508, 28.9579086, -9.6251431, 29.1392040, -38.5032120, 38.3537903
4: -16.6220894, 25.3623142, -16.7272720, 25.5436211, -42.0491104, 41.9739227
5: -7.4266629, 29.0235519, -7.4889612, 29.1675434, -36.1700172, 36.0846252
6: -38.2439575, 11.9975061, -38.3233948, 12.0483952, -50.2923508, 50.3209000
7: -11.0897322, 28.6595020, -11.2479591, 28.7649956, -38.6498184, 38.6914749
8: -21.2380238, 29.8613338, -21.3764286, 30.0560112, -50.8763962, 50.8001938
9: -13.7396574, 28.3185387, -13.9987917, 28.3509598, -42.0906181, 42.3173294
10: -22.1115913, 31.9775581, -22.5665340, 31.9864597, -54.0925293, 54.5440903
11: -23.7198048, 14.6543884, -23.9916992, 14.6401844, -38.3599892, 38.6460876
12: -44.2574463, 4.3536520, -44.5947418, 4.3360748, -45.0412521, 45.4364319
13: -37.4708710, 22.3032398, -37.5587158, 22.3515282, -59.4529724, 59.6116638
14: -64.8919525, 2.6357460, -65.3181458, 2.5710707, -67.4630203, 67.9538879
15: -21.8314896, 20.3517914, -21.8640518, 20.4577637, -42.2892532, 42.2158432
16: -23.4511471, 21.7139053, -23.7330627, 21.7034721, -45.1546173, 45.4469681
17: -58.4062271, -1.2381277, -58.7167740, -1.2720394, -55.8835907, 56.3004646
18: -35.8581161, 14.6270046, -36.0346451, 14.7115660, -50.5696831, 50.6616516
19: -26.4425964, 9.4670534, -26.6023884, 9.5443001, -35.9868965, 36.0694427
20: -21.5509357, 15.8694038, -21.6948128, 15.9211330, -37.4720688, 37.5642166
21: -27.2900982, 12.9454670, -27.5236168, 13.0089750, -40.2990723, 40.4690857
22: -32.1132698, 10.6104288, -32.1931305, 10.7415619, -42.8548317, 42.8035583
23: -24.5963383, 14.0184383, -24.7102089, 14.0814800, -38.6778183, 38.7286453
24: -30.7512627, 13.7280531, -30.7920551, 13.8047504, -44.5560150, 44.5201073
25: -28.9042625, 12.9088545, -28.9859772, 13.0319757, -41.9362373, 41.8948326
26: -41.0368652, 17.0171509, -41.2521057, 17.0959778, -58.1328430, 58.2692566
27: -26.0964031, 18.1858292, -26.1406460, 18.3232765, -44.4196777, 44.3264771
28: -25.0815887, 17.3011551, -25.1431179, 17.4085350, -42.4901237, 42.4442749
29: -27.6176929, 10.9163723, -27.7411442, 10.9861202, -38.4233093, 38.4793205
30: -26.8562775, 18.3064919, -26.9519825, 18.3567543, -45.2130318, 45.2584763
31: -35.4164124, 12.1004763, -35.6041718, 12.1955948, -47.6120071, 47.7046471
32: -35.2351379, 10.9882803, -35.3643723, 11.0213366, -45.7787704, 45.9099426
33: -63.6838951, -3.7650127, -63.6920052, -3.5340524, -55.5511169, 55.2706909
34: -57.8179932, -6.3748684, -57.8232651, -6.1488466, -47.7818756, 47.5033035
35: -56.0813560, -4.3546343, -56.0971413, -4.1133032, -45.1068420, 44.8105164
36: -53.4880676, 0.8620605, -53.5166779, 1.0243206, -49.5379410, 49.3901825
37: -78.2771149, -14.2826805, -78.3398743, -14.2547207, -60.7920837, 60.8041534
38: -63.8204918, 0.3860836, -63.8703842, 0.6012449, -59.8266907, 59.6294098
39: -72.1360016, -8.1800213, -72.1922684, -8.0645638, -58.0653992, 58.0255356
40: -51.3664665, -6.2222085, -51.4756699, -6.1427460, -45.2237206, 45.2534599
41: -40.0579834, 12.2482605, -40.1056137, 12.3093204, -52.3673019, 52.3538742
42: -26.1704044, 11.9387426, -26.2397118, 11.9730330, -38.1434364, 38.1784554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=260, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1400

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_B1_B2_B2_B1

### Relational analysis result of IS_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1479479, upper bound: 20.1324433
time: 49.91 seconds

## Relational analysis of IS_B1_B2_B2_B2

### Relational analysis result of IS_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881897, upper bound: 20.1324433
time: 49.53 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -37.6238976, 17.5717812, -37.7869415, 17.6849670, -55.3088646, 55.3587227
1: -11.9061813, 22.4692745, -12.0358181, 22.6702728, -34.5764542, 34.5050926
2: -9.6690607, 25.2752514, -9.8205633, 25.5312386, -35.2002983, 35.0958138
3: -9.5235291, 28.9481926, -9.6830425, 29.2746544, -38.5751648, 38.4041176
4: -16.5588684, 25.3547211, -16.7527771, 25.6980553, -42.1416016, 41.9920197
5: -7.3679924, 29.0126476, -7.5392194, 29.3086128, -36.2561340, 36.1220474
6: -38.2312851, 11.9679012, -38.4148369, 12.0841856, -50.3154716, 50.3827362
7: -11.0512447, 28.6532249, -11.2508888, 28.8171768, -38.6716843, 38.6852722
8: -21.1818848, 29.8553543, -21.3670597, 30.1687698, -50.9413528, 50.7844772
9: -13.7191467, 28.3007946, -13.9462061, 28.4120865, -42.1312332, 42.2470016
10: -22.0923805, 31.9350052, -22.5696926, 32.0835609, -54.1735153, 54.5046997
11: -23.6994419, 14.5842094, -24.2344532, 14.7444267, -38.4438705, 38.8186646
12: -44.2463646, 4.2379723, -44.8508148, 4.5418711, -45.2441101, 45.5581589
13: -37.4613495, 22.2507305, -37.6071968, 22.4815483, -59.6063156, 59.5934677
14: -64.8719788, 2.5295086, -65.5135117, 2.7758503, -67.6478271, 68.0430222
15: -21.7635384, 20.3338089, -21.9636936, 20.6388512, -42.4023895, 42.2975006
16: -23.4266396, 21.6812744, -23.7563953, 21.7632580, -45.1898956, 45.4376678
17: -58.3966408, -1.3316641, -58.9349403, -1.0872889, -56.0670700, 56.4212799
18: -35.8462372, 14.6095448, -36.0888824, 14.6846046, -50.5308418, 50.6984253
19: -26.4316959, 9.4307079, -26.7184391, 9.5309238, -35.9626198, 36.1491470
20: -21.5340862, 15.8290644, -21.8123398, 15.9288893, -37.4629745, 37.6414032
21: -27.2760296, 12.8950281, -27.6834335, 13.0356693, -40.3116989, 40.5784607
22: -32.0968628, 10.5931568, -32.2549133, 10.7099876, -42.8068504, 42.8480682
23: -24.5865746, 13.9896984, -24.8135300, 14.0831165, -38.6696930, 38.8032303
24: -30.7351017, 13.7205868, -30.8490391, 13.7711840, -44.5062866, 44.5696259
25: -28.8907261, 12.8801622, -29.0649223, 12.9943647, -41.8850899, 41.9450836
26: -41.0203133, 16.9632378, -41.3847961, 17.1182690, -58.1385803, 58.3480339
27: -26.0635815, 18.1792259, -26.2252922, 18.2984486, -44.3620300, 44.4045181
28: -25.0694160, 17.2729778, -25.2318401, 17.3582363, -42.4276505, 42.5048180
29: -27.6092815, 10.8790245, -27.8300304, 10.9975071, -38.4266357, 38.5336914
30: -26.8403225, 18.2803764, -27.0348873, 18.3548126, -45.1951370, 45.3152618
31: -35.4024658, 12.0598526, -35.7277298, 12.1629467, -47.5654144, 47.7875824
32: -35.2214050, 10.9437523, -35.4790726, 11.0814457, -45.8308411, 45.9813004
33: -63.6443787, -3.7847199, -63.7777939, -3.4826007, -55.5859222, 55.3434219
34: -57.7837639, -6.3929443, -57.9153976, -6.1095772, -47.8082581, 47.5862885
35: -56.0713196, -4.3674259, -56.1316032, -4.1429443, -45.0792236, 44.8482666
36: -53.4805450, 0.8286018, -53.6146507, 0.9949770, -49.5003204, 49.4655991
37: -78.2587280, -14.3175688, -78.4475555, -14.1827555, -60.8283386, 60.8848495
38: -63.8110657, 0.3453789, -63.9752769, 0.5520248, -59.7635803, 59.7294540
39: -72.1188202, -8.2044821, -72.2843094, -8.0216751, -58.0896149, 58.1243057
40: -51.3468018, -6.2287726, -51.5117798, -6.0661950, -45.2806053, 45.2830086
41: -40.0431213, 12.2286739, -40.1727638, 12.3402367, -52.3833580, 52.4014359
42: -26.1576271, 11.9133177, -26.3220310, 12.0252666, -38.1828918, 38.2353477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 941

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 733

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0923029, upper bound: 20.1895819
time: 57.14 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0923029, upper bound: 20.1895819
time: 57.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 116.72 seconds
IS_B1_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 116.72
Output dim: 5, lower bound: -20.1479479, upper bound: 20.1324433
IS_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 116.72
Output dim: 5, lower bound: -20.1881897, upper bound: 20.1324433
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 116.72
Output dim: 5, lower bound: -20.0923029, upper bound: 20.1895819
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 116.72
Output dim: 5, lower bound: -20.0923029, upper bound: 20.1895819

## BFS IS instance: IS_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -37.6583824, 17.5841331, -37.8498383, 17.6216488, -55.2800293, 55.4339714
1: -11.9353085, 22.4733887, -12.0750170, 22.6103516, -34.5456619, 34.5484047
2: -9.7182055, 25.2801895, -9.8115921, 25.4349995, -35.1532059, 35.0917816
3: -9.5865803, 28.9575272, -9.6247177, 29.1365318, -38.4930573, 38.3529510
4: -16.6219978, 25.3619957, -16.7268028, 25.5414925, -42.0402222, 41.9731445
5: -7.4265957, 29.0232296, -7.4884572, 29.1653366, -36.1583939, 36.0837898
6: -38.2438431, 11.9974499, -38.3225632, 12.0479927, -50.2918358, 50.3200150
7: -11.0896502, 28.6592598, -11.2475214, 28.7630844, -38.6356277, 38.6907539
8: -21.2379341, 29.8610287, -21.3758907, 30.0540714, -50.8648376, 50.7993851
9: -13.7395515, 28.3182259, -13.9981766, 28.3489437, -42.0884933, 42.3164024
10: -22.1114788, 31.9774837, -22.5657082, 31.9857559, -54.0963593, 54.5431900
11: -23.7194958, 14.6543407, -23.9896164, 14.6398487, -38.3593445, 38.6439590
12: -44.2572174, 4.3536043, -44.5931244, 4.3355999, -45.0404892, 45.4077988
13: -37.4708023, 22.3026924, -37.5582237, 22.3478546, -59.4492645, 59.6379242
14: -64.8915863, 2.6357098, -65.3153687, 2.5707989, -67.4623871, 67.9510803
15: -21.8314247, 20.3515282, -21.8635902, 20.4559689, -42.2873917, 42.2151184
16: -23.4510307, 21.7137623, -23.7321739, 21.7023010, -45.1533318, 45.4459381
17: -58.4058762, -1.2382507, -58.7143936, -1.2728834, -55.8824310, 56.2538834
18: -35.8578606, 14.6269073, -36.0327492, 14.7108955, -50.5687561, 50.6596565
19: -26.4422646, 9.4670029, -26.6000690, 9.5440102, -35.9862747, 36.0670700
20: -21.5506268, 15.8693590, -21.6927681, 15.9209251, -37.4715500, 37.5621262
21: -27.2897930, 12.9454527, -27.5214806, 13.0087862, -40.2985802, 40.4669342
22: -32.1128845, 10.6104031, -32.1904716, 10.7411327, -42.8540192, 42.8008728
23: -24.5959568, 14.0183811, -24.7075863, 14.0811033, -38.6770592, 38.7259674
24: -30.7507420, 13.7279873, -30.7885189, 13.8043737, -44.5551147, 44.5165062
25: -28.9038239, 12.9088068, -28.9828739, 13.0315428, -41.9353676, 41.8916817
26: -41.0365143, 17.0170898, -41.2498856, 17.0956001, -58.1321144, 58.2669754
27: -26.0960999, 18.1857681, -26.1385384, 18.3230228, -44.4191208, 44.3243065
28: -25.0812473, 17.3011055, -25.1407433, 17.4081955, -42.4894409, 42.4418488
29: -27.6172600, 10.9163218, -27.7381668, 10.9857998, -38.4225235, 38.4721451
30: -26.8558846, 18.3064537, -26.9492302, 18.3563843, -45.2122688, 45.2556839
31: -35.4159470, 12.1004333, -35.6011124, 12.1951942, -47.6111412, 47.7015457
32: -35.2348976, 10.9882145, -35.3626556, 11.0210361, -45.7781830, 45.9078217
33: -63.6836090, -3.7650609, -63.6901703, -3.5344305, -55.5503616, 55.2592926
34: -57.8177757, -6.3749132, -57.8218498, -6.1492100, -47.7812424, 47.4769058
35: -56.0812149, -4.3546572, -56.0961571, -4.1136427, -45.1062698, 44.7893982
36: -53.4878426, 0.8620071, -53.5150452, 1.0240459, -49.5373154, 49.3715210
37: -78.2768021, -14.2827291, -78.3375244, -14.2550840, -60.7913818, 60.7724152
38: -63.8201180, 0.3860044, -63.8678741, 0.6006708, -59.8256531, 59.6013947
39: -72.1356049, -8.1800900, -72.1893158, -8.0648737, -58.0645447, 58.0121460
40: -51.3662949, -6.2222986, -51.4744530, -6.1434574, -45.2228394, 45.2521553
41: -40.0578537, 12.2482529, -40.1046829, 12.3090248, -52.3668785, 52.3529358
42: -26.1702061, 11.9387035, -26.2385044, 11.9727001, -38.1429062, 38.1772079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=259, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1402
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1374

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_B1_B2_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1466601, upper bound: 20.0922016
time: 53.98 seconds

## Relational analysis of IS_B1_B2_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1466601, upper bound: 20.1324433
time: 50.63 seconds

## BFS IS instance: IS_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -37.6039734, 17.5357285, -37.7458725, 17.6104794, -55.2144547, 55.2816010
1: -11.8978300, 22.4414444, -12.0184269, 22.6134892, -34.5113182, 34.4598694
2: -9.6615400, 25.2440948, -9.8052073, 25.4677887, -35.1293297, 35.0493011
3: -9.5173187, 28.9270973, -9.6703749, 29.2313023, -38.5222778, 38.3683777
4: -16.5489693, 25.3328953, -16.7321434, 25.6531544, -42.0845718, 41.9484406
5: -7.3610559, 28.9834328, -7.5250077, 29.2490482, -36.1870346, 36.0773582
6: -38.2201843, 11.9543667, -38.3921547, 12.0564804, -50.2766647, 50.3465195
7: -11.0434952, 28.6175137, -11.2347698, 28.7433167, -38.5882263, 38.6322746
8: -21.1701488, 29.8253651, -21.3429108, 30.1067905, -50.8654938, 50.7292023
9: -13.7082968, 28.2681961, -13.9238615, 28.3449097, -42.0532074, 42.1920586
10: -22.0825710, 31.9035244, -22.5495014, 32.0184746, -54.0945740, 54.4530258
11: -23.6757526, 14.5752592, -24.1866703, 14.7257404, -38.4014931, 38.7619286
12: -44.2396698, 4.2231455, -44.8370895, 4.5116386, -45.2018280, 45.5249405
13: -37.4529800, 22.2189827, -37.5898170, 22.4165974, -59.4998474, 59.5246964
14: -64.8549271, 2.4979115, -65.4783554, 2.7107639, -67.5656891, 67.9762650
15: -21.7507439, 20.3263321, -21.9371548, 20.6235008, -42.3742447, 42.2634888
16: -23.4110756, 21.6376591, -23.7243595, 21.6734543, -45.0845299, 45.3620186
17: -58.3850021, -1.3525381, -58.9105606, -1.1304359, -56.0032654, 56.3694458
18: -35.8224907, 14.6018829, -36.0406570, 14.6689386, -50.4914284, 50.6425400
19: -26.3916149, 9.4263611, -26.6359768, 9.5220222, -35.9136353, 36.0623398
20: -21.4981728, 15.8222570, -21.7383347, 15.9149466, -37.4131203, 37.5605927
21: -27.2359238, 12.8894215, -27.6010399, 13.0242310, -40.2601547, 40.4904633
22: -32.0401764, 10.5862675, -32.1379547, 10.6960545, -42.7362289, 42.7242203
23: -24.5547104, 13.9829216, -24.7482834, 14.0690317, -38.6237411, 38.7312050
24: -30.6931019, 13.7133265, -30.7627125, 13.7563686, -44.4494705, 44.4760399
25: -28.8407288, 12.8718987, -28.9617996, 12.9773445, -41.8180733, 41.8336983
26: -40.9776497, 16.9567280, -41.2969437, 17.1048851, -58.0825348, 58.2536697
27: -26.0217171, 18.1738377, -26.1392097, 18.2872543, -44.3089714, 44.3130493
28: -25.0226097, 17.2665863, -25.1354027, 17.3452091, -42.3678207, 42.4019890
29: -27.5581188, 10.8721256, -27.7244415, 10.9834290, -38.3599014, 38.4184113
30: -26.8131580, 18.2708435, -26.9796925, 18.3353500, -45.1485062, 45.2505341
31: -35.3542404, 12.0516872, -35.6286201, 12.1461201, -47.5003586, 47.6803055
32: -35.2054367, 10.9358177, -35.4462280, 11.0650826, -45.7951965, 45.9368286
33: -63.6166763, -3.7945476, -63.7207489, -3.5027895, -55.5375061, 55.2762756
34: -57.7466888, -6.4000311, -57.8391685, -6.1243792, -47.7531738, 47.4994736
35: -56.0302505, -4.3740616, -56.0474167, -4.1566477, -45.0236206, 44.7549057
36: -53.4361458, 0.8237934, -53.5228920, 0.9849672, -49.4439087, 49.3659592
37: -78.2313843, -14.3240976, -78.3912964, -14.1962881, -60.7798767, 60.8108978
38: -63.7493591, 0.3336101, -63.8477783, 0.5275536, -59.6723557, 59.5827103
39: -72.0869751, -8.2125807, -72.2188568, -8.0383492, -58.0381088, 58.0471039
40: -51.3237228, -6.2404222, -51.4643097, -6.0900860, -45.2336349, 45.2238884
41: -40.0260315, 12.2222080, -40.1376648, 12.3268051, -52.3528366, 52.3598709
42: -26.1461182, 11.9042015, -26.2983913, 12.0064230, -38.1525421, 38.2025909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=260, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_B2_B2_A1_B1_B1

### Relational analysis result of IS_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0506693, upper bound: 20.1881895
time: 55.85 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2

### Relational analysis result of IS_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0909113, upper bound: 20.1881895
time: 53.64 seconds

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -37.6176453, 17.5655136, -37.9628067, 17.6923332, -55.3099785, 55.5283203
1: -11.9033499, 22.4664993, -12.1669197, 22.6740742, -34.5774231, 34.6334190
2: -9.6657448, 25.2725468, -9.9331360, 25.5360298, -35.2017746, 35.2056808
3: -9.5205154, 28.9460316, -9.7637234, 29.2835350, -38.5790863, 38.4832993
4: -16.5545654, 25.3522110, -16.8766651, 25.7033157, -42.1414185, 42.1133461
5: -7.3642154, 29.0095406, -7.6253481, 29.3160114, -36.2570496, 36.2052498
6: -38.2285004, 11.9624357, -38.4324226, 12.1341705, -50.3626709, 50.3948593
7: -11.0472946, 28.6502094, -11.3840542, 28.8204231, -38.6688538, 38.8145981
8: -21.1777782, 29.8524933, -21.5246239, 30.1808109, -50.9426880, 50.9384766
9: -13.7160110, 28.2969398, -14.0666170, 28.4158211, -42.1318321, 42.3635559
10: -22.0890026, 31.9299164, -22.6889973, 32.0975647, -54.1798019, 54.6189117
11: -23.6948929, 14.5807972, -24.2845516, 14.7938175, -38.4887085, 38.8653488
12: -44.2444458, 4.2334547, -44.8842468, 4.5949507, -45.2862701, 45.6052399
13: -37.4559708, 22.2460594, -37.6751099, 22.5041847, -59.5835876, 59.6928558
14: -64.8641205, 2.5244713, -65.6080170, 2.7909565, -67.6550751, 68.1324921
15: -21.7598820, 20.3313980, -22.0334568, 20.6686993, -42.4285812, 42.3648529
16: -23.4224968, 21.6770630, -23.8873539, 21.7693329, -45.1918297, 45.5644150
17: -58.3912277, -1.3349218, -59.0157890, -1.0691566, -56.0696106, 56.5293732
18: -35.8419342, 14.6068878, -36.1148071, 14.7678404, -50.6097755, 50.7216949
19: -26.4275780, 9.4282475, -26.7379913, 9.6295290, -36.0571060, 36.1662369
20: -21.5292721, 15.8261261, -21.8233509, 16.0091171, -37.5383911, 37.6494751
21: -27.2710667, 12.8920822, -27.7145824, 13.1247635, -40.3958282, 40.6066666
22: -32.0918274, 10.5904350, -32.2750130, 10.8197823, -42.9116096, 42.8654480
23: -24.5824718, 13.9866791, -24.8263245, 14.1550655, -38.7375374, 38.8130035
24: -30.7304420, 13.7171154, -30.8596687, 13.8433056, -44.5737457, 44.5767822
25: -28.8864288, 12.8772688, -29.0747547, 13.1145735, -42.0010033, 41.9520226
26: -41.0138626, 16.9595718, -41.4065399, 17.2149944, -58.2288589, 58.3661118
27: -26.0587330, 18.1761646, -26.2455883, 18.3850880, -44.4438210, 44.4217529
28: -25.0646343, 17.2697449, -25.2419758, 17.4772263, -42.5418625, 42.5117188
29: -27.6047268, 10.8762140, -27.8550758, 11.0813313, -38.5053406, 38.5556679
30: -26.8355179, 18.2770081, -27.0521622, 18.4141083, -45.2496262, 45.3291702
31: -35.3977394, 12.0564861, -35.7497559, 12.2868443, -47.6845856, 47.8062439
32: -35.2185516, 10.9407616, -35.4975891, 11.1277695, -45.8696594, 45.9989624
33: -63.6389618, -3.7880235, -63.7971878, -3.4052634, -55.6586761, 55.3530960
34: -57.7779541, -6.3954535, -57.9234734, -5.9995327, -47.9122696, 47.5837631
35: -56.0667648, -4.3699989, -56.1419868, -4.0329351, -45.1854782, 44.8445015
36: -53.4756432, 0.8266706, -53.6244202, 1.1280565, -49.6315613, 49.4609451
37: -78.2541122, -14.3201122, -78.4671631, -14.1533976, -60.8512421, 60.8927078
38: -63.8031807, 0.3422127, -63.9896049, 0.7281799, -59.9324341, 59.7175980
39: -72.1108932, -8.2075834, -72.3043671, -7.9649048, -58.1373215, 58.1335220
40: -51.3423157, -6.2378049, -51.5616150, -6.0436821, -45.2986336, 45.3238106
41: -40.0400925, 12.2247095, -40.1873550, 12.3721733, -52.4122658, 52.4120636
42: -26.1549492, 11.9075708, -26.3333626, 12.0460873, -38.2010345, 38.2409325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=260, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_B2_B2_A1_B2_B1

### Relational analysis result of IS_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0506693, upper bound: 20.1881895
time: 52.83 seconds

## Relational analysis of IS_B2_B2_A1_B2_B2

### Relational analysis result of IS_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0506693, upper bound: 20.1881895
time: 53.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 109.10 seconds
IS_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 109.10
Output dim: 5, lower bound: -20.1466601, upper bound: 20.0922016
IS_B1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 109.10
Output dim: 5, lower bound: -20.1466601, upper bound: 20.1324433
IS_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 109.10
Output dim: 5, lower bound: -20.0506693, upper bound: 20.1881895
IS_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 109.10
Output dim: 5, lower bound: -20.0909113, upper bound: 20.1881895
IS_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 109.10
Output dim: 5, lower bound: -20.0506693, upper bound: 20.1881895
IS_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 109.10
Output dim: 5, lower bound: -20.0506693, upper bound: 20.1881895

## BFS IS instance: IS_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -37.5836143, 17.5219383, -37.6689224, 17.5748920, -55.1585083, 55.1908607
1: -11.8919048, 22.4088020, -11.9736862, 22.5356903, -34.4275970, 34.3824883
2: -9.6555471, 25.2159367, -9.7705822, 25.4005661, -35.0561142, 34.9865189
3: -9.5103951, 28.8668938, -9.6219635, 29.0892353, -38.3723831, 38.2593384
4: -16.5419064, 25.2845993, -16.6844006, 25.5385914, -41.9626694, 41.8525085
5: -7.3534660, 28.9340553, -7.4800687, 29.1325798, -36.0626450, 35.9833908
6: -38.2039108, 11.9467163, -38.3495674, 12.0314693, -50.2353821, 50.2962837
7: -11.0377235, 28.5749931, -11.1905050, 28.6431465, -38.4821854, 38.5454178
8: -21.1614552, 29.7828293, -21.2856579, 30.0048332, -50.7532654, 50.6285553
9: -13.6967144, 28.2236900, -13.8656044, 28.2411633, -41.9378777, 42.0892944
10: -22.0681458, 31.8904362, -22.5033073, 31.9787292, -54.0346146, 54.3937454
11: -23.6302471, 14.5688515, -24.0716095, 14.6876469, -38.3178940, 38.6404610
12: -44.2037964, 4.2162275, -44.7488670, 4.4737864, -45.1270294, 45.4297562
13: -37.4443283, 22.1769733, -37.5548897, 22.3162346, -59.3790588, 59.4205399
14: -64.7998734, 2.4940414, -65.3394318, 2.6824389, -67.4823151, 67.8334732
15: -21.7441635, 20.2863922, -21.8986263, 20.5254784, -42.2696419, 42.1850204
16: -23.3942451, 21.6137562, -23.6588840, 21.6148148, -45.0090599, 45.2726402
17: -58.3345947, -1.3676605, -58.7796173, -1.1960583, -55.8849335, 56.2199287
18: -35.7830086, 14.5919685, -35.9466019, 14.6201248, -50.4031334, 50.5385704
19: -26.3410492, 9.4208813, -26.5149269, 9.4843674, -35.8254166, 35.9358063
20: -21.4535904, 15.8184967, -21.6307335, 15.8871317, -37.3407211, 37.4492302
21: -27.1903610, 12.8866405, -27.4887714, 12.9998751, -40.1902351, 40.3754120
22: -31.9817505, 10.5792789, -31.9983253, 10.6574898, -42.6392403, 42.5776062
23: -24.4961510, 13.9770412, -24.6083717, 14.0256519, -38.5218048, 38.5854111
24: -30.6148567, 13.7061768, -30.5771084, 13.7106962, -44.3255539, 44.2832870
25: -28.7712364, 12.8658485, -28.7960720, 12.9325285, -41.7037659, 41.6619186
26: -40.9295578, 16.9501743, -41.1797791, 17.0586662, -57.9882240, 58.1299515
27: -25.9786472, 18.1688499, -26.0379219, 18.2629948, -44.2416420, 44.2067719
28: -24.9711685, 17.2608871, -25.0130882, 17.3047256, -42.2758942, 42.2739754
29: -27.4930611, 10.8663292, -27.5686913, 10.9483795, -38.2595673, 38.2561188
30: -26.7549801, 18.2646275, -26.8386917, 18.2984409, -45.0534210, 45.1033173
31: -35.2877121, 12.0445976, -35.4697342, 12.0971794, -47.3848915, 47.5143318
32: -35.1815147, 10.9298229, -35.3857498, 11.0446320, -45.7489090, 45.8687286
33: -63.5987549, -3.8018088, -63.6739388, -3.5320635, -55.4872437, 55.2225494
34: -57.7167511, -6.4074221, -57.7672920, -6.1631117, -47.6826859, 47.4196091
35: -56.0116158, -4.3796978, -56.0024834, -4.1878872, -44.9679337, 44.7056198
36: -53.4121246, 0.8195133, -53.4669533, 0.9611702, -49.3951111, 49.3078690
37: -78.1815186, -14.3293180, -78.2711639, -14.2276745, -60.6965942, 60.6843185
38: -63.7153664, 0.3236589, -63.7670250, 0.4835854, -59.5923996, 59.4909668
39: -72.0540390, -8.2176275, -72.1378250, -8.0535383, -57.9882355, 57.9622269
40: -51.3017120, -6.2478461, -51.4054260, -6.1148982, -45.1868134, 45.1575813
41: -40.0123291, 12.2163067, -40.1015320, 12.3059492, -52.3182793, 52.3178406
42: -26.1304970, 11.8988695, -26.2561531, 11.9886665, -38.1191635, 38.1550217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_B2_B2_A1_B1_B1_B1

### Relational analysis result of IS_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
time: 54.72 seconds

## Relational analysis of IS_B2_B2_A1_B1_B1_B2

### Relational analysis result of IS_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0479773, upper bound: 20.1861014
time: 63.04 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -37.6038284, 17.5356083, -37.7448120, 17.6096230, -55.2134514, 55.2804184
1: -11.8977690, 22.4412422, -12.0180817, 22.6120052, -34.5097733, 34.4593239
2: -9.6615086, 25.2439079, -9.8048515, 25.4664955, -35.1280060, 35.0487595
3: -9.5172253, 28.9267406, -9.6699495, 29.2286148, -38.5121002, 38.3675499
4: -16.5488834, 25.3325996, -16.7316875, 25.6510258, -42.0756989, 41.9476547
5: -7.3609891, 28.9830894, -7.5245075, 29.2468414, -36.1754189, 36.0765457
6: -38.2200661, 11.9543324, -38.3913078, 12.0561094, -50.2761765, 50.3456421
7: -11.0434322, 28.6172314, -11.2343655, 28.7414188, -38.5740204, 38.6315460
8: -21.1700592, 29.8250790, -21.3423748, 30.1048489, -50.8539581, 50.7284012
9: -13.7082062, 28.2679119, -13.9232559, 28.3429012, -42.0511093, 42.1911697
10: -22.0824585, 31.9034328, -22.5487003, 32.0177422, -54.0983734, 54.4521332
11: -23.6754589, 14.5752001, -24.1846199, 14.7254257, -38.4008865, 38.7598190
12: -44.2394485, 4.2231073, -44.8354950, 4.5110731, -45.2010727, 45.4963303
13: -37.4529190, 22.2184563, -37.5893326, 22.4128456, -59.4961548, 59.5509186
14: -64.8545227, 2.4978638, -65.4756470, 2.7104445, -67.5649643, 67.9735107
15: -21.7507076, 20.3260632, -21.9367447, 20.6217289, -42.3724365, 42.2628098
16: -23.4109306, 21.6374779, -23.7234688, 21.6722584, -45.0831909, 45.3609467
17: -58.3846855, -1.3526707, -58.9082336, -1.1312733, -56.0021362, 56.3228416
18: -35.8222084, 14.6018085, -36.0387535, 14.6682587, -50.4904671, 50.6405640
19: -26.3912735, 9.4263191, -26.6336422, 9.5217361, -35.9130096, 36.0599594
20: -21.4978867, 15.8222370, -21.7362518, 15.9147606, -37.4126472, 37.5584869
21: -27.2356167, 12.8893938, -27.5989189, 13.0240612, -40.2596779, 40.4883118
22: -32.0397873, 10.5861940, -32.1352539, 10.6956711, -42.7354584, 42.7214470
23: -24.5543461, 13.9828672, -24.7456360, 14.0686874, -38.6230316, 38.7285042
24: -30.6925964, 13.7132559, -30.7591629, 13.7559376, -44.4485321, 44.4724197
25: -28.8402672, 12.8718128, -28.9587040, 12.9769430, -41.8172112, 41.8305168
26: -40.9773560, 16.9566669, -41.2947006, 17.1044521, -58.0818100, 58.2513657
27: -26.0214214, 18.1737900, -26.1370792, 18.2870178, -44.3084412, 44.3108673
28: -25.0222607, 17.2665443, -25.1330299, 17.3448849, -42.3671455, 42.3995743
29: -27.5576897, 10.8721104, -27.7214813, 10.9831152, -38.3591309, 38.4112396
30: -26.8127899, 18.2707977, -26.9769650, 18.3349876, -45.1477776, 45.2477646
31: -35.3537979, 12.0515728, -35.6255646, 12.1456566, -47.4994545, 47.6771393
32: -35.2052002, 10.9357595, -35.4445190, 11.0647602, -45.7946091, 45.9347534
33: -63.6164246, -3.7946019, -63.7189255, -3.5031595, -55.5367432, 55.2649765
34: -57.7464981, -6.4000816, -57.8377228, -6.1247826, -47.7525024, 47.4731064
35: -56.0301056, -4.3740768, -56.0464249, -4.1569529, -45.0230865, 44.7338409
36: -53.4359169, 0.8237534, -53.5212517, 0.9846954, -49.4432983, 49.3473511
37: -78.2310562, -14.3241615, -78.3889923, -14.1966143, -60.7791748, 60.7792130
38: -63.7489929, 0.3335238, -63.8452530, 0.5270066, -59.6713028, 59.5547256
39: -72.0865326, -8.2125769, -72.2158890, -8.0386839, -58.0372925, 58.0337296
40: -51.3235855, -6.2405281, -51.4631119, -6.0907497, -45.2328339, 45.2225838
41: -40.0258789, 12.2221699, -40.1367188, 12.3264790, -52.3523560, 52.3588867
42: -26.1459312, 11.9041548, -26.2971821, 12.0060778, -38.1520081, 38.2013359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1374

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0909114, upper bound: 20.1479476
time: 67.07 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0909114, upper bound: 20.1881895
time: 57.71 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -37.5972824, 17.5517330, -37.8858643, 17.6567955, -55.2540779, 55.4375992
1: -11.8974295, 22.4338284, -12.1222143, 22.5962791, -34.4937096, 34.5560417
2: -9.6597633, 25.2443924, -9.8985281, 25.4688015, -35.1285629, 35.1429214
3: -9.5135975, 28.8858032, -9.7152882, 29.1414146, -38.4291916, 38.3742828
4: -16.5475311, 25.3039112, -16.8289757, 25.5887089, -42.0195160, 42.0174866
5: -7.3566222, 28.9601631, -7.5803909, 29.1994343, -36.1326294, 36.1113129
6: -38.2122269, 11.9547596, -38.3897705, 12.1092911, -50.3215179, 50.3445282
7: -11.0415382, 28.6076984, -11.3398037, 28.7202263, -38.5628738, 38.7277946
8: -21.1690903, 29.8099689, -21.4673710, 30.0788498, -50.8305283, 50.8378677
9: -13.7044697, 28.2524338, -14.0084410, 28.3120899, -42.0165596, 42.2608757
10: -22.0745792, 31.9168339, -22.6427822, 32.0577164, -54.1197891, 54.5596161
11: -23.6493683, 14.5743923, -24.1695538, 14.7557821, -38.4051514, 38.7439461
12: -44.2085876, 4.2266092, -44.7960968, 4.5570955, -45.2115021, 45.5100479
13: -37.4472656, 22.2040043, -37.6401978, 22.4039078, -59.4628296, 59.5887756
14: -64.8090210, 2.5206079, -65.4691925, 2.7626791, -67.5717010, 67.9897995
15: -21.7532768, 20.2914486, -21.9949760, 20.5706863, -42.3239632, 42.2864227
16: -23.4056778, 21.6531906, -23.8218231, 21.7107162, -45.1163940, 45.4750137
17: -58.3408165, -1.3499718, -58.8849068, -1.1346951, -55.9512253, 56.3799133
18: -35.8024902, 14.5970135, -36.0207329, 14.7191105, -50.5215988, 50.6177444
19: -26.3770218, 9.4227676, -26.6169319, 9.5918598, -35.9688797, 36.0396996
20: -21.4846916, 15.8223343, -21.7157745, 15.9812794, -37.4659729, 37.5381088
21: -27.2255211, 12.8893156, -27.6024017, 13.1003866, -40.3259087, 40.4917183
22: -32.0333405, 10.5834332, -32.1353531, 10.7812376, -42.8145790, 42.7187881
23: -24.5238972, 13.9807625, -24.6864662, 14.1117496, -38.6356468, 38.6672287
24: -30.6522102, 13.7099533, -30.6740780, 13.7976551, -44.4498672, 44.3840332
25: -28.8169289, 12.8711987, -28.9090042, 13.0697784, -41.8867073, 41.7802048
26: -40.9657478, 16.9529114, -41.2894516, 17.1687660, -58.1345139, 58.2423630
27: -26.0156441, 18.1711960, -26.1442547, 18.3608475, -44.3764915, 44.3154526
28: -25.0131989, 17.2640591, -25.1196995, 17.4367332, -42.4499321, 42.3837585
29: -27.5396461, 10.8703947, -27.6993332, 11.0463486, -38.4050827, 38.3934021
30: -26.7773418, 18.2707977, -26.9111691, 18.3772087, -45.1545486, 45.1819687
31: -35.3311920, 12.0494576, -35.5909195, 12.2379713, -47.5691643, 47.6403770
32: -35.1946259, 10.9347744, -35.4370880, 11.1073895, -45.8232880, 45.9308472
33: -63.6210327, -3.7953167, -63.7500877, -3.4345360, -55.6083832, 55.2991409
34: -57.7480469, -6.4028912, -57.8515892, -6.0382299, -47.8418121, 47.5038757
35: -56.0481491, -4.3756399, -56.0969696, -4.0642118, -45.1297684, 44.7950859
36: -53.4515686, 0.8223391, -53.5684128, 1.1042614, -49.5827332, 49.4028091
37: -78.2042694, -14.3253727, -78.3467941, -14.1847725, -60.7679596, 60.7659988
38: -63.7691574, 0.3323531, -63.9088287, 0.6842875, -59.8524933, 59.6258316
39: -72.0778732, -8.2126856, -72.2230682, -7.9800968, -58.0875397, 58.0483322
40: -51.3203430, -6.2452841, -51.5027313, -6.0684829, -45.2518616, 45.2574463
41: -40.0263977, 12.2188272, -40.1511497, 12.3513031, -52.3777008, 52.3699760
42: -26.1393585, 11.9022551, -26.2910843, 12.0283871, -38.1677475, 38.1933403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_B2_B2_A1_B2_B1_B1

### Relational analysis result of IS_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0503886, upper bound: 20.1861014
time: 62.36 seconds

## Relational analysis of IS_B2_B2_A1_B2_B1_B2

### Relational analysis result of IS_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0901138, upper bound: 20.1861014
time: 56.84 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -37.6175232, 17.5653801, -37.9617119, 17.6914654, -55.3089905, 55.5270920
1: -11.9032879, 22.4662781, -12.1665745, 22.6725864, -34.5758743, 34.6328506
2: -9.6656914, 25.2723579, -9.9327803, 25.5347481, -35.2004395, 35.2051392
3: -9.5204420, 28.9456367, -9.7632523, 29.2808495, -38.5689240, 38.4824600
4: -16.5545082, 25.3519211, -16.8762054, 25.7011852, -42.1325531, 42.1125717
5: -7.3641276, 29.0092163, -7.6248307, 29.3137894, -36.2454605, 36.2044716
6: -38.2283859, 11.9623899, -38.4315643, 12.1338139, -50.3621979, 50.3939552
7: -11.0472488, 28.6499424, -11.3836517, 28.8184929, -38.6546555, 38.8139381
8: -21.1776791, 29.8522072, -21.5240669, 30.1788578, -50.9311752, 50.9376602
9: -13.7159462, 28.2966385, -14.0659628, 28.4137955, -42.1297417, 42.3626022
10: -22.0888672, 31.9298306, -22.6882343, 32.0968475, -54.1836548, 54.6180649
11: -23.6945915, 14.5807533, -24.2824917, 14.7934742, -38.4880676, 38.8632431
12: -44.2442284, 4.2333841, -44.8826103, 4.5944271, -45.2854996, 45.5765839
13: -37.4558640, 22.2455235, -37.6746025, 22.5004520, -59.5798645, 59.7191086
14: -64.8637543, 2.5244331, -65.6052551, 2.7906847, -67.6544418, 68.1296844
15: -21.7598228, 20.3311272, -22.0330200, 20.6669025, -42.4267273, 42.3641472
16: -23.4223614, 21.6768990, -23.8864574, 21.7681770, -45.1905365, 45.5633545
17: -58.3909607, -1.3350468, -59.0134163, -1.0700293, -56.0684509, 56.4827385
18: -35.8416595, 14.6067791, -36.1128883, 14.7671556, -50.6088142, 50.7196655
19: -26.4272480, 9.4281979, -26.7356453, 9.6292229, -36.0564728, 36.1638412
20: -21.5289726, 15.8260889, -21.8212585, 16.0088844, -37.5378571, 37.6473465
21: -27.2707596, 12.8920765, -27.7124844, 13.1246071, -40.3953667, 40.6045609
22: -32.0914459, 10.5903702, -32.2723236, 10.8194008, -42.9108467, 42.8626938
23: -24.5820808, 13.9866447, -24.8236942, 14.1547146, -38.7367935, 38.8103409
24: -30.7299404, 13.7170467, -30.8561268, 13.8428526, -44.5727921, 44.5731735
25: -28.8859863, 12.8772116, -29.0716324, 13.1141510, -42.0001373, 41.9488449
26: -41.0135689, 16.9595108, -41.4043503, 17.2145901, -58.2281570, 58.3638611
27: -26.0584278, 18.1761379, -26.2434731, 18.3848324, -44.4432602, 44.4196091
28: -25.0643196, 17.2696991, -25.2396030, 17.4768696, -42.5411911, 42.5093002
29: -27.6042595, 10.8761473, -27.8520851, 11.0809650, -38.5046463, 38.5484848
30: -26.8351059, 18.2769585, -27.0494900, 18.4137611, -45.2488670, 45.3264465
31: -35.3972931, 12.0564308, -35.7467155, 12.2864208, -47.6837158, 47.8031464
32: -35.2182999, 10.9407215, -35.4958725, 11.1274652, -45.8690414, 45.9968033
33: -63.6387024, -3.7880979, -63.7952995, -3.4056978, -55.6579437, 55.3417969
34: -57.7777519, -6.3954964, -57.9220390, -5.9999189, -47.9116669, 47.5573807
35: -56.0666313, -4.3700657, -56.1409645, -4.0332260, -45.1849060, 44.8234329
36: -53.4753876, 0.8265734, -53.6228027, 1.1278152, -49.6309967, 49.4423447
37: -78.2537689, -14.3201876, -78.4648590, -14.1537275, -60.8505096, 60.8610535
38: -63.8027916, 0.3421640, -63.9870529, 0.7275963, -59.9314346, 59.6896210
39: -72.1104965, -8.2076321, -72.3014145, -7.9652386, -58.1365356, 58.1201706
40: -51.3421249, -6.2379117, -51.5603867, -6.0443640, -45.2977600, 45.3224754
41: -40.0399628, 12.2246714, -40.1864548, 12.3718376, -52.4118004, 52.4111252
42: -26.1547985, 11.9075089, -26.3321400, 12.0457335, -38.2005310, 38.2396469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1374

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0909114, upper bound: 20.1479471
time: 64.16 seconds

## Relational analysis of IS_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324436, upper bound: 20.1881890
time: 54.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 121.33 seconds
IS_B2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
IS_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0479773, upper bound: 20.1861014
IS_B2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0909114, upper bound: 20.1479476
IS_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0909114, upper bound: 20.1881895
IS_B2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0503886, upper bound: 20.1861014
IS_B2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0901138, upper bound: 20.1861014
IS_B2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.0909114, upper bound: 20.1479471
IS_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 121.33
Output dim: 5, lower bound: -20.1324436, upper bound: 20.1881890

## BFS IS instance: IS_B2_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -37.5808525, 17.5164871, -37.6573029, 17.5521660, -55.1330185, 55.1737900
1: -11.8907785, 22.4032135, -11.9689474, 22.5122604, -34.4030380, 34.3721619
2: -9.6545677, 25.2099075, -9.7664423, 25.3752975, -35.0298653, 34.9763489
3: -9.5095997, 28.8626022, -9.6187038, 29.0713329, -38.3531418, 38.2517471
4: -16.5404224, 25.2798958, -16.6780491, 25.5189476, -41.9411469, 41.8414345
5: -7.3524928, 28.9289703, -7.4758568, 29.1112766, -36.0399666, 35.9740448
6: -38.1995354, 11.9449196, -38.3317757, 12.0239067, -50.2234421, 50.2766953
7: -11.0367184, 28.5679321, -11.1863384, 28.6135120, -38.4516296, 38.5341187
8: -21.1598091, 29.7757721, -21.2788582, 29.9752350, -50.7216797, 50.6145782
9: -13.6953964, 28.2173729, -13.8598881, 28.2146797, -41.9100761, 42.0772629
10: -22.0667419, 31.8842258, -22.4974728, 31.9528198, -54.0062790, 54.3816986
11: -23.6275673, 14.5676222, -24.0606880, 14.6825209, -38.3100891, 38.6283112
12: -44.2006226, 4.2140598, -44.7363586, 4.4645052, -45.1141968, 45.4146881
13: -37.4431953, 22.1742191, -37.5501862, 22.3047523, -59.3566742, 59.4095306
14: -64.7972565, 2.4865971, -65.3284149, 2.6513414, -67.4486008, 67.8150101
15: -21.7420998, 20.2856331, -21.8899002, 20.5223503, -42.2644501, 42.1755333
16: -23.3925133, 21.6053734, -23.6516685, 21.5795841, -44.9720993, 45.2570419
17: -58.3327179, -1.3702106, -58.7717171, -1.2068739, -55.8646088, 56.2076912
18: -35.7787895, 14.5911331, -35.9289093, 14.6166306, -50.3954201, 50.5200424
19: -26.3354492, 9.4205589, -26.4915409, 9.4828453, -35.8182945, 35.9121017
20: -21.4474602, 15.8175564, -21.6050606, 15.8832245, -37.3306847, 37.4226151
21: -27.1845856, 12.8858538, -27.4646606, 12.9965057, -40.1810913, 40.3505135
22: -31.9738140, 10.5785580, -31.9652786, 10.6544914, -42.6283035, 42.5438385
23: -24.4922428, 13.9760151, -24.5920563, 14.0213146, -38.5135574, 38.5680695
24: -30.6105328, 13.7052946, -30.5593796, 13.7069960, -44.3175278, 44.2646751
25: -28.7638721, 12.8648062, -28.7653198, 12.9282045, -41.6920776, 41.6301270
26: -40.9213943, 16.9492455, -41.1456909, 17.0549622, -57.9763565, 58.0949364
27: -25.9751930, 18.1681290, -26.0233459, 18.2599010, -44.2350922, 44.1914749
28: -24.9651184, 17.2599602, -24.9877911, 17.3006077, -42.2657242, 42.2477493
29: -27.4867134, 10.8654881, -27.5421829, 10.9449568, -38.2495880, 38.2282181
30: -26.7530041, 18.2631683, -26.8304501, 18.2923698, -45.0453720, 45.0936203
31: -35.2808228, 12.0437164, -35.4409065, 12.0934229, -47.3742447, 47.4846230
32: -35.1776352, 10.9285507, -35.3695450, 11.0393620, -45.7393951, 45.8511200
33: -63.5912476, -3.8028231, -63.6429596, -3.5362697, -55.4752579, 55.1906204
34: -57.7095032, -6.4083128, -57.7368393, -6.1668148, -47.6712952, 47.3877945
35: -56.0027924, -4.3803215, -55.9654694, -4.1906128, -44.9562378, 44.6675568
36: -53.4014893, 0.8187952, -53.4224663, 0.9582586, -49.3812714, 49.2616425
37: -78.1730881, -14.3301487, -78.2358017, -14.2311611, -60.6834106, 60.6457520
38: -63.6995850, 0.3221684, -63.7009659, 0.4773712, -59.5701523, 59.4226913
39: -72.0458679, -8.2183352, -72.1045990, -8.0565720, -57.9765625, 57.9275665
40: -51.2986336, -6.2488670, -51.3922729, -6.1191831, -45.1794510, 45.1434059
41: -40.0077515, 12.2152386, -40.0823212, 12.3012972, -52.3090477, 52.2975616
42: -26.1267910, 11.8974819, -26.2406940, 11.9828424, -38.1096344, 38.1381760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B2_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0181258, upper bound: 20.1506388
time: 50.53 seconds

## Relational analysis of IS_B2_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0181258, upper bound: 20.1847981
time: 53.94 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -37.5796661, 17.5182629, -37.8833389, 17.5960484, -55.1757126, 55.4016037
1: -11.8899364, 22.4066734, -12.1319418, 22.5435104, -34.4334488, 34.5386162
2: -9.6533785, 25.2137451, -9.9073925, 25.4057579, -35.0591354, 35.1211395
3: -9.5087719, 28.8650742, -9.7430210, 29.0996418, -38.3804016, 38.3788834
4: -16.5389652, 25.2822533, -16.8513451, 25.5451241, -41.9662018, 42.0171204
5: -7.3516555, 28.9318848, -7.6171284, 29.1394157, -36.0672913, 36.1182480
6: -38.2005615, 11.9429636, -38.3674622, 12.1486044, -50.3491669, 50.3104248
7: -11.0354929, 28.5722198, -11.3690376, 28.6481171, -38.4851303, 38.7217827
8: -21.1590881, 29.7801132, -21.5001297, 30.0194244, -50.7639771, 50.8408356
9: -13.6947746, 28.2212963, -14.0125713, 28.2476387, -41.9424133, 42.2338676
10: -22.0661182, 31.8873863, -22.6581211, 32.0001335, -54.0533142, 54.5455093
11: -23.6271667, 14.5666885, -24.1237240, 14.7445126, -38.3716812, 38.6904144
12: -44.2013054, 4.2137499, -44.7681885, 4.6239367, -45.2692413, 45.4461823
13: -37.4407997, 22.1749344, -37.6085815, 22.3506374, -59.3817139, 59.5205688
14: -64.7942810, 2.4892063, -65.4846039, 2.7004061, -67.4946899, 67.9738083
15: -21.7418861, 20.2820740, -22.0089149, 20.5434303, -42.2853165, 42.2909889
16: -23.3916779, 21.6105232, -23.8134022, 21.6302128, -45.0218887, 45.4239273
17: -58.3306808, -1.3694687, -58.8763809, -1.1685514, -55.8951874, 56.3624840
18: -35.7809105, 14.5904846, -35.9779358, 14.7241745, -50.5050850, 50.5684204
19: -26.3385143, 9.4194374, -26.5406857, 9.5812092, -35.9197235, 35.9601212
20: -21.4506969, 15.8166447, -21.6534748, 15.9762430, -37.4269409, 37.4701195
21: -27.1873550, 12.8849449, -27.5310364, 13.0901127, -40.2774658, 40.4159813
22: -31.9782543, 10.5779982, -32.0262909, 10.7576818, -42.7359352, 42.6042900
23: -24.4938984, 13.9750843, -24.6275558, 14.1024933, -38.5963898, 38.6026382
24: -30.6112118, 13.7045822, -30.5926743, 13.7709351, -44.3821487, 44.2972565
25: -28.7679977, 12.8642921, -28.8120689, 13.0439167, -41.8119125, 41.6763611
26: -40.9258499, 16.9480934, -41.2105637, 17.1868782, -58.1127281, 58.1586571
27: -25.9762039, 18.1674480, -26.0645180, 18.3238010, -44.3000031, 44.2319641
28: -24.9683723, 17.2590523, -25.0318241, 17.4201221, -42.3884964, 42.2908783
29: -27.4896221, 10.8648491, -27.5981712, 11.0236835, -38.3315582, 38.2850113
30: -26.7524662, 18.2626801, -26.8663216, 18.3385506, -45.0910187, 45.1290016
31: -35.2847595, 12.0427465, -35.5007782, 12.2242107, -47.5089722, 47.5435257
32: -35.1774406, 10.9272957, -35.3995590, 11.1448231, -45.8439102, 45.8797836
33: -63.5953369, -3.8035712, -63.7000961, -3.3873544, -55.6286850, 55.2460480
34: -57.7137833, -6.4090290, -57.7763023, -6.0285358, -47.8169479, 47.4255066
35: -56.0080719, -4.3809156, -56.0119934, -4.0396404, -45.1136322, 44.7116699
36: -53.4083939, 0.8183384, -53.4794693, 1.1473560, -49.5765076, 49.3146133
37: -78.1775513, -14.3310146, -78.2951660, -14.1074219, -60.8119507, 60.7056885
38: -63.7101707, 0.3217969, -63.7892799, 0.7341528, -59.8350983, 59.5016479
39: -72.0496063, -8.2190495, -72.1649628, -7.9309244, -58.1042328, 57.9861069
40: -51.2992554, -6.2504787, -51.4403000, -6.0364146, -45.2628403, 45.1898193
41: -40.0102463, 12.2132282, -40.1214142, 12.3992910, -52.4095383, 52.3346405
42: -26.1279335, 11.8948784, -26.2765884, 12.0682955, -38.1962280, 38.1714668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B2_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0454527, upper bound: 20.1506383
time: 53.40 seconds

## Relational analysis of IS_B2_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0181258, upper bound: 20.1847987
time: 57.23 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -37.6029434, 17.5348568, -37.7448120, 17.6096230, -55.2125664, 55.2796707
1: -11.8975086, 22.4399815, -12.0180817, 22.6120052, -34.5095139, 34.4580612
2: -9.6611872, 25.2427940, -9.8048515, 25.4664955, -35.1276817, 35.0476456
3: -9.5168495, 28.9244423, -9.6699495, 29.2286148, -38.5117111, 38.3578339
4: -16.5484924, 25.3307610, -16.7316875, 25.6510258, -42.0752945, 41.9391632
5: -7.3605309, 28.9812012, -7.5245075, 29.2468414, -36.1749992, 36.0653343
6: -38.2193298, 11.9540205, -38.3913078, 12.0561094, -50.2754402, 50.3453293
7: -11.0430984, 28.6155930, -11.2343655, 28.7414188, -38.5736618, 38.6176949
8: -21.1695786, 29.8234272, -21.3423748, 30.1048489, -50.8535004, 50.7172241
9: -13.7076674, 28.2661839, -13.9232559, 28.3429012, -42.0505676, 42.1894379
10: -22.0818176, 31.9028034, -22.5487003, 32.0177422, -54.0975189, 54.4515038
11: -23.6737080, 14.5749168, -24.1846199, 14.7254257, -38.3991318, 38.7595367
12: -44.2380524, 4.2226067, -44.8354950, 4.5110731, -45.1727829, 45.4958954
13: -37.4524765, 22.2152481, -37.5893326, 22.4128456, -59.5237732, 59.5485458
14: -64.8521423, 2.4976282, -65.4756470, 2.7104445, -67.5625839, 67.9732742
15: -21.7503223, 20.3245430, -21.9367447, 20.6217289, -42.3720512, 42.2612877
16: -23.4101830, 21.6364498, -23.7234688, 21.6722584, -45.0824432, 45.3599167
17: -58.3826904, -1.3534508, -58.9082336, -1.1312733, -55.9559174, 56.3220940
18: -35.8205872, 14.6011820, -36.0387535, 14.6682587, -50.4888458, 50.6399345
19: -26.3892975, 9.4260969, -26.6336422, 9.5217361, -35.9110336, 36.0597382
20: -21.4960823, 15.8220539, -21.7362518, 15.9147606, -37.4108429, 37.5583038
21: -27.2337627, 12.8892603, -27.5989189, 13.0240612, -40.2578239, 40.4881783
22: -32.0375061, 10.5858917, -32.1352539, 10.6956711, -42.7331772, 42.7211456
23: -24.5520859, 13.9825668, -24.7456360, 14.0686874, -38.6207733, 38.7282028
24: -30.6895466, 13.7129402, -30.7591629, 13.7559376, -44.4454842, 44.4721031
25: -28.8375778, 12.8714972, -28.9587040, 12.9769430, -41.8145218, 41.8302002
26: -40.9754524, 16.9563370, -41.2947006, 17.1044521, -58.0799026, 58.2510376
27: -26.0196114, 18.1735802, -26.1370792, 18.2870178, -44.3066292, 44.3106613
28: -25.0202484, 17.2662582, -25.1330299, 17.3448849, -42.3651352, 42.3992882
29: -27.5551434, 10.8718128, -27.7214813, 10.9831152, -38.3524323, 38.4109879
30: -26.8104362, 18.2704735, -26.9769650, 18.3349876, -45.1454239, 45.2474365
31: -35.3511887, 12.0512390, -35.6255646, 12.1456566, -47.4968452, 47.6768036
32: -35.2037277, 10.9354954, -35.4445190, 11.0647602, -45.7928009, 45.9344330
33: -63.6148338, -3.7949224, -63.7189255, -3.5031595, -55.5256271, 55.2645874
34: -57.7452698, -6.4004087, -57.8377228, -6.1247826, -47.7263947, 47.4727478
35: -56.0292282, -4.3743582, -56.0464249, -4.1569529, -45.0021057, 44.7335587
36: -53.4345245, 0.8235188, -53.5212517, 0.9846954, -49.4249268, 49.3471146
37: -78.2290573, -14.3244390, -78.3889923, -14.1966143, -60.7478638, 60.7788696
38: -63.7468567, 0.3330069, -63.8452530, 0.5270066, -59.6438370, 59.5542450
39: -72.0840378, -8.2128906, -72.2158890, -8.0386839, -58.0243378, 58.0334091
40: -51.3225021, -6.2411103, -51.4631119, -6.0907497, -45.2317505, 45.2220001
41: -40.0250931, 12.2218819, -40.1367188, 12.3264790, -52.3515701, 52.3586006
42: -26.1448936, 11.9038763, -26.2971821, 12.0060778, -38.1509705, 38.2010574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_B2_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
time: 43.83 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0479773, upper bound: 20.1861014
time: 309.14 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -37.5944748, 17.5462780, -37.8742599, 17.6339722, -55.2284470, 55.4205399
1: -11.8963194, 22.4282131, -12.1174641, 22.5728226, -34.4691429, 34.5456772
2: -9.6587687, 25.2383537, -9.8943949, 25.4435387, -35.1023064, 35.1327477
3: -9.5128078, 28.8814888, -9.7120543, 29.1235313, -38.4098969, 38.3666534
4: -16.5460110, 25.2992020, -16.8227081, 25.5689964, -41.9978943, 42.0064545
5: -7.3556237, 28.9550495, -7.5762100, 29.1780300, -36.1098404, 36.1019516
6: -38.2078094, 11.9529428, -38.3721008, 12.1018410, -50.3096504, 50.3250427
7: -11.0405788, 28.6006203, -11.3356514, 28.6905594, -38.5322189, 38.7164955
8: -21.1674881, 29.8028946, -21.4606018, 30.0492516, -50.7988968, 50.8239441
9: -13.7031193, 28.2461052, -14.0027809, 28.2855167, -41.9886360, 42.2488861
10: -22.0732002, 31.9106674, -22.6369705, 32.0317841, -54.0913544, 54.5476379
11: -23.6466637, 14.5731735, -24.1584034, 14.7507010, -38.3973656, 38.7315750
12: -44.2053833, 4.2243872, -44.7843781, 4.5478363, -45.1985703, 45.4950790
13: -37.4461479, 22.2012711, -37.6355476, 22.3922501, -59.4403229, 59.5778732
14: -64.8063965, 2.5131245, -65.4582138, 2.7314205, -67.5378189, 67.9713364
15: -21.7512226, 20.2906990, -21.9864349, 20.5675602, -42.3187828, 42.2771339
16: -23.4039345, 21.6447906, -23.8146362, 21.6754093, -45.0793457, 45.4594269
17: -58.3389282, -1.3525763, -58.8770752, -1.1455736, -55.9309235, 56.3676338
18: -35.7982254, 14.5961828, -36.0030365, 14.7156258, -50.5138512, 50.5992203
19: -26.3714123, 9.4224091, -26.5935574, 9.5903854, -35.9617996, 36.0159683
20: -21.4785595, 15.8214264, -21.6900558, 15.9773149, -37.4558754, 37.5114822
21: -27.2197342, 12.8885002, -27.5782356, 13.0970278, -40.3167610, 40.4667358
22: -32.0254211, 10.5827246, -32.1022110, 10.7781582, -42.8035812, 42.6849365
23: -24.5199776, 13.9797554, -24.6699772, 14.1074581, -38.6274338, 38.6497345
24: -30.6478348, 13.7090549, -30.6562481, 13.7939491, -44.4417839, 44.3653030
25: -28.8095779, 12.8701954, -28.8782578, 13.0654316, -41.8750076, 41.7484512
26: -40.9575729, 16.9520664, -41.2552986, 17.1650429, -58.1226158, 58.2073669
27: -26.0121593, 18.1704655, -26.1294708, 18.3577766, -44.3699341, 44.2999344
28: -25.0071297, 17.2630882, -25.0943890, 17.4326401, -42.4397697, 42.3574753
29: -27.5333099, 10.8695507, -27.6727695, 11.0428619, -38.3950348, 38.3654251
30: -26.7753258, 18.2693367, -26.9027214, 18.3711605, -45.1464844, 45.1720581
31: -35.3242531, 12.0485344, -35.5620651, 12.2342319, -47.5584869, 47.6105995
32: -35.1907463, 10.9335041, -35.4208832, 11.1021442, -45.8139267, 45.9134369
33: -63.6135025, -3.7963309, -63.7186890, -3.4387450, -55.5963898, 55.2670364
34: -57.7407341, -6.4037600, -57.8209496, -6.0419016, -47.8304672, 47.4719391
35: -56.0393219, -4.3762922, -56.0599518, -4.0668602, -45.1180496, 44.7570038
36: -53.4409447, 0.8216696, -53.5238724, 1.1013670, -49.5689468, 49.3565750
37: -78.1958160, -14.3262033, -78.3111801, -14.1882257, -60.7548218, 60.7271881
38: -63.7533722, 0.3308654, -63.8426933, 0.6781197, -59.8303375, 59.5575333
39: -72.0697556, -8.2133579, -72.1892776, -7.9831257, -58.0758743, 58.0135193
40: -51.3172150, -6.2463241, -51.4895477, -6.0728106, -45.2444038, 45.2432251
41: -40.0218239, 12.2177391, -40.1318359, 12.3467445, -52.3685684, 52.3495750
42: -26.1356659, 11.9008579, -26.2756977, 12.0225544, -38.1582184, 38.1765556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B2_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0478609, upper bound: 20.1506383
time: 52.56 seconds

## Relational analysis of IS_B2_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0478609, upper bound: 20.1847981
time: 292.88 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -37.5933533, 17.5484924, -38.1007347, 17.6787167, -55.2720718, 55.6492271
1: -11.8955116, 22.4317303, -12.2810249, 22.6041222, -34.4996338, 34.7127533
2: -9.6575985, 25.2422104, -10.0356865, 25.4739227, -35.1315231, 35.2778969
3: -9.5119934, 28.8839493, -9.8361826, 29.1518497, -38.4372253, 38.4937248
4: -16.5445747, 25.3015747, -16.9967041, 25.5949745, -42.0228271, 42.1829529
5: -7.3548036, 28.9579620, -7.7171860, 29.2063770, -36.1373749, 36.2459488
6: -38.2088852, 11.9510489, -38.4077110, 12.2276974, -50.4365845, 50.3587608
7: -11.0393667, 28.6049290, -11.5188866, 28.7252159, -38.5658569, 38.9047432
8: -21.1668453, 29.8072548, -21.6824989, 30.0935535, -50.8414001, 51.0508575
9: -13.7025156, 28.2500572, -14.1554623, 28.3186531, -42.0211678, 42.4055176
10: -22.0726032, 31.9138165, -22.7980957, 32.0791550, -54.1385803, 54.7119141
11: -23.6463413, 14.5722866, -24.2224541, 14.8134680, -38.4598083, 38.7947388
12: -44.2060928, 4.2241249, -44.8105011, 4.7077179, -45.3542480, 45.5270233
13: -37.4437675, 22.2020569, -37.6944046, 22.4383926, -59.4657440, 59.6897736
14: -64.8034821, 2.5158091, -65.6146317, 2.7813931, -67.5848770, 68.1304398
15: -21.7510681, 20.2871056, -22.1068726, 20.5890083, -42.3400764, 42.3939781
16: -23.4031448, 21.6499634, -23.9768753, 21.7262611, -45.1294060, 45.6268387
17: -58.3369408, -1.3518162, -58.9847412, -1.1068172, -55.9619446, 56.5246735
18: -35.8003311, 14.5955667, -36.0521889, 14.8232956, -50.6236267, 50.6477547
19: -26.3744621, 9.4212751, -26.6427765, 9.6887379, -36.0632019, 36.0640526
20: -21.4818153, 15.8205242, -21.7384872, 16.0711079, -37.5529251, 37.5590134
21: -27.2225094, 12.8876524, -27.6446533, 13.1914749, -40.4139862, 40.5323067
22: -32.0299072, 10.5822029, -32.1635056, 10.8815031, -42.9114113, 42.7457085
23: -24.5216732, 13.9788666, -24.7058296, 14.1892204, -38.7108917, 38.6846962
24: -30.6485100, 13.7084084, -30.6898422, 13.8581486, -44.5066605, 44.3982506
25: -28.8137131, 12.8697386, -28.9250526, 13.1817255, -41.9954376, 41.7947922
26: -40.9620667, 16.9508896, -41.3203506, 17.2970047, -58.2590714, 58.2712402
27: -26.0132332, 18.1698151, -26.1708755, 18.4218731, -44.4351044, 44.3406906
28: -25.0104370, 17.2621803, -25.1383247, 17.5521679, -42.5626068, 42.4005051
29: -27.5362358, 10.8689184, -27.7291126, 11.1218281, -38.4773407, 38.4225616
30: -26.7748260, 18.2689266, -26.9385948, 18.4181633, -45.1929893, 45.2075195
31: -35.3282242, 12.0476084, -35.6220322, 12.3654919, -47.6937180, 47.6696396
32: -35.1905594, 10.9322634, -35.4510269, 11.2081776, -45.9190369, 45.9419174
33: -63.6175919, -3.7970061, -63.7763176, -3.2891545, -55.7507019, 55.3227997
34: -57.7451057, -6.4044781, -57.8607368, -5.9029999, -47.9768524, 47.5099258
35: -56.0445862, -4.3768759, -56.1065216, -3.9153404, -45.2761307, 44.8012619
36: -53.4478378, 0.8211880, -53.5807114, 1.2909088, -49.7647781, 49.4094315
37: -78.2003784, -14.3270588, -78.3710175, -14.0641899, -60.8838806, 60.7878113
38: -63.7639465, 0.3304815, -63.9304886, 0.9356551, -60.0961609, 59.6361542
39: -72.0735092, -8.2139988, -72.2502289, -7.8566647, -58.2047729, 58.0725327
40: -51.3178787, -6.2478614, -51.5388947, -5.9914236, -45.3264542, 45.2910347
41: -40.0243683, 12.2157869, -40.1710930, 12.4449959, -52.4693642, 52.3868790
42: -26.1368313, 11.8982601, -26.3114243, 12.1085472, -38.2453766, 38.2096863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=311, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1416
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B2_B2_A1_B2_B1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0875833, upper bound: 20.1506383
time: 69.16 seconds

## Relational analysis of IS_B2_B2_A1_B2_B1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0875833, upper bound: 20.1847987
time: 49.67 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -37.6165924, 17.5646343, -37.9617119, 17.6914654, -55.3080597, 55.5263443
1: -11.9030132, 22.4650040, -12.1665745, 22.6725864, -34.5755997, 34.6315765
2: -9.6654110, 25.2712517, -9.9327803, 25.5347481, -35.2001572, 35.2040329
3: -9.5200596, 28.9433193, -9.7632523, 29.2808495, -38.5685425, 38.4727631
4: -16.5540791, 25.3500862, -16.8762054, 25.7011852, -42.1321487, 42.1040878
5: -7.3636718, 29.0073204, -7.6248307, 29.3137894, -36.2450562, 36.1932526
6: -38.2276421, 11.9620733, -38.4315643, 12.1338139, -50.3614578, 50.3936386
7: -11.0468893, 28.6482849, -11.3836517, 28.8184929, -38.6542969, 38.8000832
8: -21.1772614, 29.8505192, -21.5240669, 30.1788578, -50.9307175, 50.9264908
9: -13.7153997, 28.2949181, -14.0659628, 28.4137955, -42.1291962, 42.3608818
10: -22.0882244, 31.9292011, -22.6882343, 32.0968475, -54.1827240, 54.6174355
11: -23.6928444, 14.5804567, -24.2824917, 14.7934742, -38.4863205, 38.8629494
12: -44.2428169, 4.2329712, -44.8826103, 4.5944271, -45.2572250, 45.5761261
13: -37.4554558, 22.2423096, -37.6746025, 22.5004520, -59.6074753, 59.7167816
14: -64.8613892, 2.5241871, -65.6052551, 2.7906847, -67.6520767, 68.1294403
15: -21.7594414, 20.3296070, -22.0330200, 20.6669025, -42.4263458, 42.3626251
16: -23.4216061, 21.6758862, -23.8864574, 21.7681770, -45.1897812, 45.5623436
17: -58.3889503, -1.3357544, -59.0134163, -1.0700293, -56.0222397, 56.4819870
18: -35.8400421, 14.6062269, -36.1128883, 14.7671556, -50.6071968, 50.7191162
19: -26.4252434, 9.4279480, -26.7356453, 9.6292229, -36.0544662, 36.1635933
20: -21.5272141, 15.8259153, -21.8212585, 16.0088844, -37.5360985, 37.6471748
21: -27.2689400, 12.8919210, -27.7124844, 13.1246071, -40.3935471, 40.6044044
22: -32.0891647, 10.5900688, -32.2723236, 10.8194008, -42.9085655, 42.8623924
23: -24.5798359, 13.9863167, -24.8236942, 14.1547146, -38.7345505, 38.8100128
24: -30.7268772, 13.7167091, -30.8561268, 13.8428526, -44.5697289, 44.5728378
25: -28.8833103, 12.8768682, -29.0716324, 13.1141510, -41.9974594, 41.9485016
26: -41.0116196, 16.9591713, -41.4043503, 17.2145901, -58.2262115, 58.3635216
27: -26.0565948, 18.1759224, -26.2434731, 18.3848324, -44.4414291, 44.4193954
28: -25.0622787, 17.2694016, -25.2396030, 17.4768696, -42.5391464, 42.5090027
29: -27.6017551, 10.8758869, -27.8520851, 11.0809650, -38.4979172, 38.5482254
30: -26.8327808, 18.2766533, -27.0494900, 18.4137611, -45.2465439, 45.3261414
31: -35.3946762, 12.0560646, -35.7467155, 12.2864208, -47.6810989, 47.8027802
32: -35.2168503, 10.9404469, -35.4958725, 11.1274652, -45.8672180, 45.9965134
33: -63.6371155, -3.7884083, -63.7952995, -3.4056978, -55.6467133, 55.3413925
34: -57.7765312, -6.3958263, -57.9220390, -5.9999189, -47.8855438, 47.5570221
35: -56.0657883, -4.3703527, -56.1409645, -4.0332260, -45.1638870, 44.8231277
36: -53.4739838, 0.8263512, -53.6228027, 1.1278152, -49.6125946, 49.4421005
37: -78.2517700, -14.3204708, -78.4648590, -14.1537275, -60.8192291, 60.8607635
38: -63.8006554, 0.3416586, -63.9870529, 0.7275963, -59.9038849, 59.6891022
39: -72.1079407, -8.2079515, -72.3014145, -7.9652386, -58.1235199, 58.1198273
40: -51.3411026, -6.2384996, -51.5603867, -6.0443640, -45.2967377, 45.3218880
41: -40.0391693, 12.2244015, -40.1864548, 12.3718376, -52.4110069, 52.4108582
42: -26.1537533, 11.9072332, -26.3321400, 12.0457335, -38.1994858, 38.2393723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_B2_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
time: 194.21 seconds

## Relational analysis of IS_B2_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861014
time: 62.74 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 259.27 seconds
IS_B2_B2_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0181258, upper bound: 20.1506388
IS_B2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0181258, upper bound: 20.1847981
IS_B2_B2_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0454527, upper bound: 20.1506383
IS_B2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0181258, upper bound: 20.1847987
IS_B2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
IS_B2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0479773, upper bound: 20.1861014
IS_B2_B2_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0478609, upper bound: 20.1506383
IS_B2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0478609, upper bound: 20.1847981
IS_B2_B2_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0875833, upper bound: 20.1506383
IS_B2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0875833, upper bound: 20.1847987
IS_B2_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
IS_B2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 259.27
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861014

## BFS IS instance: IS_B2_B2_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -37.5801163, 17.5158768, -37.6571274, 17.5520573, -55.1321716, 55.1730042
1: -11.8905869, 22.4019432, -11.9688959, 22.5119934, -34.4025803, 34.3708382
2: -9.6543245, 25.2089310, -9.7664118, 25.3750935, -35.0294189, 34.9753418
3: -9.5093107, 28.8609047, -9.6186352, 29.0709743, -38.3524857, 38.2430801
4: -16.5400085, 25.2783451, -16.6779251, 25.5186100, -41.9404068, 41.8328476
5: -7.3520679, 28.9275684, -7.4757566, 29.1109657, -36.0392532, 35.9622726
6: -38.1982841, 11.9445620, -38.3315010, 12.0238533, -50.2221375, 50.2760620
7: -11.0364170, 28.5665207, -11.1862593, 28.6132259, -38.4510040, 38.5205574
8: -21.1593876, 29.7741089, -21.2787590, 29.9748898, -50.7209091, 50.5975800
9: -13.6949749, 28.2161560, -13.8597908, 28.2144146, -41.9093895, 42.0759468
10: -22.0662308, 31.8837547, -22.4973679, 31.9527206, -54.0046921, 54.3811226
11: -23.6266708, 14.5673838, -24.0605011, 14.6824703, -38.3091431, 38.6278839
12: -44.1987534, 4.2134686, -44.7359467, 4.4644041, -45.0809860, 45.4137115
13: -37.4396095, 22.1735611, -37.5495453, 22.3046093, -59.3587112, 59.4073410
14: -64.7957535, 2.4863644, -65.3280411, 2.6512709, -67.4470215, 67.8144073
15: -21.7416191, 20.2844181, -21.8897915, 20.5220795, -42.2636986, 42.1742096
16: -23.3918495, 21.6041527, -23.6515446, 21.5793190, -44.9711685, 45.2556992
17: -58.3307419, -1.3707237, -58.7712555, -1.2069864, -55.8147964, 56.2068520
18: -35.7784309, 14.5907059, -35.9288330, 14.6165104, -50.3949432, 50.5195389
19: -26.3340664, 9.4203997, -26.4912529, 9.4828138, -35.8168793, 35.9116516
20: -21.4461708, 15.8174057, -21.6047859, 15.8831758, -37.3293457, 37.4221916
21: -27.1832581, 12.8857231, -27.4643707, 12.9964876, -40.1797447, 40.3500938
22: -31.9720669, 10.5782967, -31.9648647, 10.6544285, -42.6264954, 42.5431595
23: -24.4908638, 13.9757195, -24.5917492, 14.0212631, -38.5121269, 38.5674667
24: -30.6089897, 13.7050495, -30.5590553, 13.7069321, -44.3159218, 44.2641068
25: -28.7621460, 12.8645258, -28.7649536, 12.9281359, -41.6902809, 41.6294785
26: -40.9204025, 16.9489098, -41.1454697, 17.0548668, -57.9752693, 58.0943794
27: -25.9743671, 18.1673660, -26.0231571, 18.2597313, -44.2341003, 44.1905212
28: -24.9638557, 17.2596779, -24.9875183, 17.3005714, -42.2644272, 42.2471962
29: -27.4848652, 10.8652945, -27.5417633, 10.9449158, -38.2444687, 38.2275848
30: -26.7516003, 18.2627926, -26.8301392, 18.2922764, -45.0438766, 45.0929337
31: -35.2792511, 12.0434942, -35.4405785, 12.0933561, -47.3726082, 47.4840736
32: -35.1761398, 10.9282417, -35.3692474, 11.0392838, -45.7329559, 45.8504639
33: -63.5898438, -3.8031354, -63.6426773, -3.5363421, -55.4474945, 55.1899490
34: -57.7084808, -6.4085846, -57.7366409, -6.1668758, -47.6413574, 47.3872757
35: -56.0015717, -4.3805008, -55.9652252, -4.1906500, -44.9177246, 44.6672134
36: -53.3996506, 0.8185749, -53.4220772, 0.9581842, -49.3450165, 49.2610703
37: -78.1709595, -14.3304958, -78.2353516, -14.2312450, -60.6489716, 60.6449585
38: -63.6976852, 0.3218236, -63.7005539, 0.4772696, -59.5205688, 59.4219284
39: -72.0437012, -8.2186365, -72.1040878, -8.0566244, -57.9360352, 57.9267731
40: -51.2977676, -6.2492027, -51.3920898, -6.1192660, -45.1785011, 45.1428871
41: -40.0068130, 12.2149086, -40.0821304, 12.3012257, -52.3080368, 52.2970390
42: -26.1259613, 11.8971653, -26.2404766, 11.9827795, -38.1087418, 38.1376419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_B2_B2_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_B2_B2_A1_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9935371, upper bound: 20.1824209
time: 76.86 seconds

## Relational analysis of IS_B2_B2_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_B2_B2_A1_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9935371, upper bound: 20.1824204
time: 67.54 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -37.5789490, 17.5176487, -37.8831787, 17.5959301, -55.1748810, 55.4008255
1: -11.8897524, 22.4054108, -12.1318913, 22.5432320, -34.4329834, 34.5373001
2: -9.6531658, 25.2127666, -9.9073181, 25.4055481, -35.0587158, 35.1200867
3: -9.5084610, 28.8633385, -9.7429581, 29.0992718, -38.3796997, 38.3702850
4: -16.5385532, 25.2806969, -16.8512383, 25.5447865, -41.9654999, 42.0085220
5: -7.3512487, 28.9304848, -7.6170511, 29.1391125, -36.0666084, 36.1064911
6: -38.1993103, 11.9426241, -38.3672180, 12.1485338, -50.3478432, 50.3098412
7: -11.0351963, 28.5708427, -11.3689728, 28.6478233, -38.4845428, 38.7082062
8: -21.1587162, 29.7784538, -21.5000420, 30.0190525, -50.7632370, 50.8238220
9: -13.6943550, 28.2200890, -14.0124807, 28.2473717, -41.9417267, 42.2325706
10: -22.0656128, 31.8869514, -22.6579952, 32.0000153, -54.0517120, 54.5449448
11: -23.6262493, 14.5664501, -24.1235294, 14.7444582, -38.3707085, 38.6899796
12: -44.1993866, 4.2131901, -44.7677765, 4.6238012, -45.2360229, 45.4452591
13: -37.4372215, 22.1742535, -37.6079216, 22.3504810, -59.3836365, 59.5182724
14: -64.7928009, 2.4889755, -65.4842834, 2.7003660, -67.4931641, 67.9732590
15: -21.7414093, 20.2808418, -22.0088139, 20.5431709, -42.2845802, 42.2896576
16: -23.3910217, 21.6093159, -23.8132324, 21.6299591, -45.0209808, 45.4225464
17: -58.3286667, -1.3700285, -58.8759613, -1.1686831, -55.8454361, 56.3616867
18: -35.7804985, 14.5900593, -35.9778671, 14.7240849, -50.5045853, 50.5679245
19: -26.3371220, 9.4192514, -26.5404072, 9.5811691, -35.9182892, 35.9596596
20: -21.4493866, 15.8164902, -21.6531906, 15.9762049, -37.4255905, 37.4696808
21: -27.1860142, 12.8848124, -27.5307655, 13.0900936, -40.2761078, 40.4155769
22: -31.9765015, 10.5777617, -32.0259094, 10.7576370, -42.7341385, 42.6036720
23: -24.4925079, 13.9747944, -24.6272411, 14.1024275, -38.5949364, 38.6020355
24: -30.6096668, 13.7043362, -30.5923729, 13.7708578, -44.3805237, 44.2967072
25: -28.7662544, 12.8640308, -28.8117027, 13.0438499, -41.8101044, 41.6757355
26: -40.9248619, 16.9478035, -41.2103806, 17.1868286, -58.1116905, 58.1581841
27: -25.9753761, 18.1667137, -26.0643768, 18.3236542, -44.2990303, 44.2310905
28: -24.9671288, 17.2588367, -25.0315552, 17.4200859, -42.3872147, 42.2903900
29: -27.4877377, 10.8646297, -27.5977783, 11.0236416, -38.3264465, 38.2844009
30: -26.7510548, 18.2623482, -26.8660202, 18.3384838, -45.0895386, 45.1283684
31: -35.2831726, 12.0425100, -35.5004463, 12.2241669, -47.5073395, 47.5429573
32: -35.1759567, 10.9269886, -35.3992233, 11.1447430, -45.8375168, 45.8791351
33: -63.5939484, -3.8038726, -63.6998138, -3.3874254, -55.6008606, 55.2453613
34: -57.7127686, -6.4093533, -57.7761040, -6.0285950, -47.7869873, 47.4250183
35: -56.0068550, -4.3811579, -56.0117340, -4.0396929, -45.0751190, 44.7113419
36: -53.4065285, 0.8180876, -53.4790726, 1.1472664, -49.5402679, 49.3140564
37: -78.1754379, -14.3313370, -78.2946777, -14.1074886, -60.7775574, 60.7049255
38: -63.7082558, 0.3214083, -63.7888794, 0.7340522, -59.7855759, 59.5008926
39: -72.0474548, -8.2193413, -72.1644592, -7.9309893, -58.0637054, 57.9853363
40: -51.2984009, -6.2507496, -51.4400864, -6.0364523, -45.2619476, 45.1893387
41: -40.0093384, 12.2129631, -40.1212349, 12.3992386, -52.4085770, 52.3341980
42: -26.1270905, 11.8945446, -26.2763405, 12.0682087, -38.1952972, 38.1708832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_B2_B2_A1_B1_B1_B2_A2_A1

### Relational analysis result of IS_B2_B2_A1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0208858, upper bound: 20.1824204
time: 52.23 seconds

## Relational analysis of IS_B2_B2_A1_B1_B1_B2_A2_A2

### Relational analysis result of IS_B2_B2_A1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9935371, upper bound: 20.1824204
time: 53.72 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -37.6001778, 17.5293999, -37.7332573, 17.5868988, -55.1870766, 55.2626572
1: -11.8963709, 22.4343910, -12.0133448, 22.5885868, -34.4849586, 34.4477348
2: -9.6602058, 25.2367706, -9.8007259, 25.4412079, -35.1014137, 35.0374985
3: -9.5160713, 28.9201469, -9.6666975, 29.2107143, -38.4924850, 38.3502502
4: -16.5469952, 25.3260651, -16.7253113, 25.6313782, -42.0537491, 41.9280891
5: -7.3595500, 28.9760914, -7.5203524, 29.2255020, -36.1523094, 36.0559731
6: -38.2149506, 11.9522305, -38.3735123, 12.0485840, -50.2635345, 50.3257446
7: -11.0420837, 28.6085167, -11.2301950, 28.7117748, -38.5430984, 38.6063766
8: -21.1679764, 29.8163452, -21.3355446, 30.0752792, -50.8218842, 50.7032471
9: -13.7063274, 28.2598381, -13.9175425, 28.3163776, -42.0227051, 42.1773796
10: -22.0804310, 31.8966522, -22.5429153, 31.9918365, -54.0690994, 54.4395676
11: -23.6710510, 14.5737114, -24.1736965, 14.7203197, -38.3913727, 38.7474060
12: -44.2349052, 4.2203636, -44.8228989, 4.5018253, -45.1599426, 45.4808121
13: -37.4513931, 22.2125282, -37.5846786, 22.4013443, -59.5013733, 59.5376282
14: -64.8495255, 2.4901972, -65.4645691, 2.6793175, -67.5288391, 67.9547653
15: -21.7482567, 20.3237820, -21.9280148, 20.6185722, -42.3668289, 42.2517967
16: -23.4084759, 21.6280384, -23.7162743, 21.6369934, -45.0454712, 45.3443146
17: -58.3808060, -1.3560009, -58.9003181, -1.1421347, -55.9355850, 56.3098488
18: -35.8163528, 14.6003532, -36.0211182, 14.6647491, -50.4811020, 50.6214714
19: -26.3836651, 9.4257145, -26.6102772, 9.5202713, -35.9039383, 36.0359917
20: -21.4899521, 15.8211174, -21.7105827, 15.9108086, -37.4007607, 37.5317001
21: -27.2279778, 12.8884392, -27.5747414, 13.0207138, -40.2486916, 40.4631805
22: -32.0295868, 10.5851316, -32.1022034, 10.6926928, -42.7222786, 42.6873360
23: -24.5481873, 13.9815531, -24.7293282, 14.0643616, -38.6125488, 38.7108803
24: -30.6852493, 13.7120495, -30.7414188, 13.7522898, -44.4375381, 44.4534683
25: -28.8302612, 12.8704748, -28.9279594, 12.9726229, -41.8028831, 41.7984352
26: -40.9672813, 16.9554405, -41.2605820, 17.1007195, -58.0680008, 58.2160225
27: -26.0161285, 18.1728268, -26.1224689, 18.2839203, -44.3000488, 44.2952957
28: -25.0142097, 17.2652702, -25.1077309, 17.3407803, -42.3549881, 42.3730011
29: -27.5487919, 10.8710098, -27.6949215, 10.9797087, -38.3424530, 38.3829727
30: -26.8084564, 18.2690792, -26.9686947, 18.3289185, -45.1373749, 45.2377739
31: -35.3443069, 12.0503464, -35.5966949, 12.1419468, -47.4862518, 47.6470413
32: -35.1998520, 10.9342499, -35.4282990, 11.0594587, -45.7833252, 45.9168243
33: -63.6073151, -3.7959046, -63.6878891, -3.5073714, -55.5136414, 55.2326584
34: -57.7379303, -6.4012842, -57.8072968, -6.1284857, -47.7150269, 47.4409561
35: -56.0204163, -4.3750143, -56.0093689, -4.1596651, -44.9903641, 44.6954575
36: -53.4238586, 0.8228903, -53.4767036, 0.9817924, -49.4110947, 49.3008804
37: -78.2206192, -14.3252678, -78.3536377, -14.2000885, -60.7347412, 60.7403030
38: -63.7310753, 0.3315377, -63.7791672, 0.5207696, -59.6216431, 59.4859543
39: -72.0758972, -8.2135782, -72.1826019, -8.0417042, -58.0126419, 57.9987259
40: -51.3193817, -6.2421360, -51.4499092, -6.0950470, -45.2243347, 45.2077713
41: -40.0204964, 12.2207813, -40.1175156, 12.3218527, -52.3423500, 52.3382950
42: -26.1412125, 11.9024611, -26.2817421, 12.0002480, -38.1414604, 38.1842041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1678

## Relational analysis of IS_B2_B2_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0004168, upper bound: 20.1847981
time: 45.68 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0181257, upper bound: 20.1847982
time: 102.86 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -37.5990028, 17.5311928, -37.9592857, 17.6307049, -55.2297058, 55.4904785
1: -11.8955326, 22.4378605, -12.1763678, 22.6198044, -34.5153351, 34.6142273
2: -9.6590309, 25.2406063, -9.9416571, 25.4716797, -35.1307106, 35.1822624
3: -9.5152588, 28.9225998, -9.7910709, 29.2390099, -38.5196915, 38.4774208
4: -16.5455360, 25.3284073, -16.8986015, 25.6575356, -42.0788574, 42.1037750
5: -7.3587313, 28.9790115, -7.6616058, 29.2536621, -36.1796150, 36.2001801
6: -38.2159882, 11.9502468, -38.4092026, 12.1732845, -50.3892746, 50.3594513
7: -11.0408535, 28.6128426, -11.4128952, 28.7463455, -38.5765686, 38.7940178
8: -21.1672611, 29.8207207, -21.5568562, 30.1194191, -50.8641891, 50.9294968
9: -13.7057095, 28.2637653, -14.0702448, 28.3493366, -42.0550461, 42.3340111
10: -22.0798054, 31.8997879, -22.7035789, 32.0392876, -54.1163025, 54.6033669
11: -23.6706200, 14.5727921, -24.2366219, 14.7822857, -38.4529037, 38.8094139
12: -44.2355690, 4.2201300, -44.8547287, 4.6612034, -45.3150101, 45.5123215
13: -37.4489975, 22.2132206, -37.6431122, 22.4471283, -59.5262985, 59.6486053
14: -64.8465958, 2.4928102, -65.6206360, 2.7283936, -67.5749893, 68.1134491
15: -21.7480354, 20.3202152, -22.0469818, 20.6396484, -42.3876839, 42.3671951
16: -23.4076157, 21.6331902, -23.8780575, 21.6875896, -45.0952072, 45.5112457
17: -58.3787918, -1.3552771, -59.0048065, -1.1037331, -55.9661484, 56.4644623
18: -35.8184586, 14.5997772, -36.0700684, 14.7723198, -50.5907784, 50.6698456
19: -26.3867245, 9.4246082, -26.6593723, 9.6185846, -36.0053101, 36.0839806
20: -21.4932060, 15.8201790, -21.7589092, 16.0038528, -37.4970589, 37.5790863
21: -27.2307243, 12.8875580, -27.6410084, 13.1143560, -40.3450813, 40.5285645
22: -32.0340424, 10.5845976, -32.1632233, 10.7958698, -42.8299103, 42.7478218
23: -24.5498676, 13.9806337, -24.7647743, 14.1454983, -38.6953659, 38.7454071
24: -30.6858768, 13.7113380, -30.7746887, 13.8162022, -44.5020790, 44.4860268
25: -28.8343658, 12.8699417, -28.9746552, 13.0883551, -41.9227219, 41.8445969
26: -40.9717178, 16.9543037, -41.3254242, 17.2326736, -58.2043915, 58.2797279
27: -26.0171509, 18.1721649, -26.1636429, 18.3478279, -44.3649788, 44.3358078
28: -25.0174942, 17.2644024, -25.1517048, 17.4602757, -42.4777679, 42.4161072
29: -27.5516891, 10.8703232, -27.7508869, 11.0583839, -38.4243622, 38.4397659
30: -26.8079205, 18.2685661, -27.0045242, 18.3751240, -45.1830444, 45.2730904
31: -35.3482437, 12.0493822, -35.6565552, 12.2727604, -47.6210022, 47.7059364
32: -35.1996689, 10.9329815, -35.4583664, 11.1649113, -45.8878479, 45.9455261
33: -63.6114159, -3.7966413, -63.7451782, -3.3584652, -55.6670837, 55.2881851
34: -57.7422714, -6.4020319, -57.8467674, -5.9902039, -47.8606796, 47.4786453
35: -56.0257149, -4.3756237, -56.0559196, -4.0087166, -45.1478043, 44.7396774
36: -53.4307861, 0.8223715, -53.5337143, 1.1708984, -49.6062927, 49.3538666
37: -78.2251053, -14.3261395, -78.4130402, -14.0763178, -60.8632507, 60.8002930
38: -63.7416573, 0.3311591, -63.8674927, 0.7775779, -59.8865509, 59.5648651
39: -72.0796432, -8.2143126, -72.2429657, -7.9160070, -58.1403351, 58.0574417
40: -51.3200455, -6.2437291, -51.4979858, -6.0123568, -45.3076897, 45.2542572
41: -40.0229988, 12.2188253, -40.1566467, 12.4198132, -52.4428101, 52.3754730
42: -26.1423187, 11.8998356, -26.3176270, 12.0856628, -38.2279816, 38.2174606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 935
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 935
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1402
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1416
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1400
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B2_B2_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0454527, upper bound: 20.1506383
time: 371.61 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0454527, upper bound: 20.1847981
time: 64.55 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 438.49 seconds
IS_B2_B2_A1_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 438.49
Output dim: 5, lower bound: -19.9935371, upper bound: 20.1824209
IS_B2_B2_A1_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 438.49
Output dim: 5, lower bound: -19.9935371, upper bound: 20.1824204
IS_B2_B2_A1_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 438.49
Output dim: 5, lower bound: -20.0208858, upper bound: 20.1824204
IS_B2_B2_A1_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 438.49
Output dim: 5, lower bound: -19.9935371, upper bound: 20.1824204
IS_B2_B2_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 438.49
Output dim: 5, lower bound: -20.0004168, upper bound: 20.1847981
IS_B2_B2_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 438.49
Output dim: 5, lower bound: -20.0181257, upper bound: 20.1847982
IS_B2_B2_A1_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 438.49
Output dim: 5, lower bound: -20.0454527, upper bound: 20.1506383
IS_B2_B2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 438.49
Output dim: 5, lower bound: -20.0454527, upper bound: 20.1847981
IS_B2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 438.49
Output dim: 5, lower bound: -20.0478609, upper bound: 20.1847981
IS_B2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 438.49
Output dim: 5, lower bound: -20.0875833, upper bound: 20.1847987
IS_B2_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 438.49
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861019
IS_B2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 438.49
Output dim: 5, lower bound: -20.0206428, upper bound: 20.1861014

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 60.07 + 3975.92 = 4035.99 seconds
