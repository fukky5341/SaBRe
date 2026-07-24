## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 79.4085105772
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612)
1: (-59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992)
2: (-53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439)
3: (-58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935)
4: (-66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492)
5: (-60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872)
6: (-95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498)
7: (-70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839)
8: (-77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492)
9: (-64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648)
10: (-95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122)
11: (-91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326)
12: (-91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001)
13: (-97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521)
14: (-141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317)
15: (-75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994)
16: (-96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474)
17: (-139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440)
18: (-90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867)
19: (-72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048)
20: (-67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908)
21: (-89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444)
22: (-90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815)
23: (-71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896)
24: (-86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379)
25: (-76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524)
26: (-101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566)
27: (-88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820)
28: (-71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006)
29: (-92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924)
30: (-88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199)
31: (-93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035)
32: (-93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315)
33: (-121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873)
34: (-100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062)
35: (-98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817)
36: (-100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760)
37: (-141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120)
38: (-119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850)
39: (-135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140)
40: (-112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437)
41: (-94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704)
42: (-69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776)

## BASE Result
execution time: IAR + LP analysis = 2.78 + 98.11 = 100.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -83.2047295, upper bound: 83.2047295


# Binary Search by BASE starts (time budget: 17899.11 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=124.9945068359375
rel_dist={1: [-79.47797671394284, 79.47797670615168]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=124.9945068359375
rel_dist={1: [-76.66888544603012, 76.66888544397699]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=124.9945068359375
rel_dist={1: [-77.72391643090478, 77.72391642674157]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=124.9945068359375
rel_dist={1: [-78.64399325943418, 78.64399325111157]}

## Binary Search Result
Binary search time: 518.65 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 17380.46 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5538443, upper bound: 81.4997863
time: 153.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5538443, upper bound: 81.5538443
time: 318.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 471.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 471.82
Output dim: 1, lower bound: -81.5538443, upper bound: 81.4997863
IS_A2, status: Status.UNKNOWN, split count: 1, time: 471.82
Output dim: 1, lower bound: -81.5538443, upper bound: 81.5538443

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -106.3987808, 78.3931427, -106.4806671, 78.6289520, -185.0277405, 184.8738098
1: -59.1949806, 65.5602417, -59.2339211, 65.7312927, -124.9262695, 124.7941589
2: -52.9987984, 63.1506500, -53.0301323, 63.3093071, -116.3080978, 116.1807861
3: -58.3475113, 72.5665359, -58.3751678, 72.7197189, -131.0672302, 130.9416962
4: -66.6753998, 73.7898865, -66.7164307, 73.9738464, -140.6492310, 140.5063171
5: -60.4376793, 70.3861237, -60.4675293, 70.5629730, -131.0006409, 130.8536530
6: -95.3362808, 62.5405197, -95.4602280, 62.5910530, -157.9273376, 158.0007477
7: -70.7053680, 67.3509064, -70.7430115, 67.5455780, -138.2509460, 138.0939178
8: -76.9942551, 89.8392029, -77.0405807, 90.0639801, -167.0582275, 166.8797913
9: -64.8290787, 71.1079102, -64.8747787, 71.2930984, -136.1221771, 135.9826965
10: -95.4483795, 90.6976547, -95.5052490, 90.8525772, -186.3009644, 186.2029114
11: -91.7407379, 56.0580711, -91.8517227, 56.0944710, -147.8352051, 147.9097900
12: -90.9319611, 72.7274628, -91.0411835, 72.7872009, -163.7191620, 163.7686462
13: -97.0572739, 95.5465546, -97.1137238, 95.7165298, -192.7738037, 192.6602631
14: -141.8136902, 80.0412674, -141.9015808, 80.1782227, -221.9919128, 221.9428406
15: -75.3955841, 67.1489944, -75.4515076, 67.2360153, -142.6315918, 142.6004944
16: -96.0858536, 69.4087601, -96.1641693, 69.5493469, -165.6351929, 165.5729370
17: -139.3055725, 76.2069168, -139.3822174, 76.3034210, -215.6089935, 215.5891418
18: -90.8292084, 75.3867340, -90.9694595, 75.4265137, -166.2557220, 166.3562012
19: -72.6127319, 45.5880623, -72.7724762, 45.6074638, -118.2201996, 118.3605347
20: -67.4992828, 52.5560455, -67.6285248, 52.5902481, -120.0895157, 120.1845703
21: -88.9300766, 53.5384483, -89.0947495, 53.5666771, -142.4967499, 142.6331940
22: -90.7209625, 55.0547943, -90.9361267, 55.0865326, -145.8074951, 145.9909210
23: -71.2746277, 55.7239151, -71.4207458, 55.7526588, -127.0272827, 127.1446609
24: -85.9065552, 51.3063354, -86.0765533, 51.3299713, -137.2365112, 137.3828888
25: -76.0526505, 58.9300232, -76.2391510, 58.9625397, -135.0151978, 135.1691742
26: -101.0170441, 82.2497406, -101.1886749, 82.2895050, -183.3065491, 183.4384155
27: -88.5054169, 59.8431320, -88.7150040, 59.8729134, -148.3783264, 148.5581360
28: -71.5487061, 60.9409828, -71.7378693, 60.9718895, -132.5205688, 132.6788483
29: -92.1360092, 48.6692810, -92.3376617, 48.6957092, -140.8317108, 141.0069275
30: -88.3624573, 65.4331512, -88.4981232, 65.4801941, -153.8426514, 153.9312744
31: -93.1065674, 55.1229706, -93.3298111, 55.1519737, -148.2585449, 148.4527893
32: -93.0414886, 59.1597519, -93.1772995, 59.1990051, -152.2404785, 152.3370361
33: -121.5373611, 78.7372742, -121.6683426, 78.7862549, -200.3235779, 200.4056091
34: -100.6080170, 57.9516373, -100.7431793, 57.9866562, -158.5946655, 158.6948242
35: -98.7382889, 60.6092262, -98.8946838, 60.6424179, -159.3806915, 159.5038757
36: -100.2701111, 62.8136902, -100.4526062, 62.8406639, -163.1107788, 163.2662964
37: -141.5543823, 61.9797630, -141.7049103, 62.0087509, -203.5631104, 203.6846771
38: -119.4981003, 78.5923767, -119.7086639, 78.6419678, -198.1400452, 198.3010101
39: -134.8947449, 75.5172653, -135.0255585, 75.5543289, -210.4490662, 210.5428162
40: -111.8822021, 61.2507858, -111.9855042, 61.2935066, -173.1757050, 173.2362976
41: -94.0716019, 64.6205139, -94.1819382, 64.6531525, -158.7247467, 158.8024597
42: -69.1534271, 59.9795609, -69.2158966, 60.0235138, -129.1769409, 129.1954651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5522923, upper bound: 81.4697241
time: 97.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5522923, upper bound: 81.4950978
time: 107.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -106.8154907, 78.6753693, -106.4872131, 78.6585007, -185.4739990, 185.1625671
1: -59.4805145, 65.7647858, -59.2370033, 65.7527695, -125.2332840, 125.0017853
2: -53.2288322, 63.3461914, -53.0320511, 63.3292236, -116.5580597, 116.3782349
3: -58.5482407, 72.7629242, -58.3768730, 72.7387924, -131.2870331, 131.1398010
4: -66.9480667, 74.0127106, -66.7190399, 73.9967651, -140.9448242, 140.7317505
5: -60.6308441, 70.6114120, -60.4689484, 70.5850449, -131.2158508, 131.0803528
6: -95.5125351, 62.7040367, -95.4751434, 62.5944748, -158.1069946, 158.1791840
7: -71.0024414, 67.5852814, -70.7453918, 67.5699921, -138.5724182, 138.3306732
8: -77.3389282, 90.1202164, -77.0441895, 90.0920105, -167.4309387, 167.1643982
9: -65.0898590, 71.3347473, -64.8788757, 71.3161926, -136.4060516, 136.2136230
10: -95.7453384, 90.9067078, -95.5102081, 90.8712921, -186.6166077, 186.4169159
11: -91.9487000, 56.1931839, -91.8640900, 56.0968323, -148.0455322, 148.0572815
12: -91.0908203, 72.9143219, -91.0540771, 72.7928696, -163.8836975, 163.9683990
13: -97.2929535, 95.7867126, -97.1170883, 95.7372742, -193.0302124, 192.9037781
14: -142.1135254, 80.2237549, -141.9078522, 80.1955261, -222.3090210, 222.1315918
15: -75.5925140, 67.2743225, -75.4564056, 67.2460022, -142.8385162, 142.7307281
16: -96.4193039, 69.5926208, -96.1715012, 69.5665436, -165.9858398, 165.7641296
17: -139.6358032, 76.3525314, -139.3875122, 76.3138885, -215.9496918, 215.7400208
18: -91.0398788, 75.6030121, -90.9862518, 75.4297943, -166.4696655, 166.5892639
19: -72.8342285, 45.7722473, -72.7920380, 45.6082382, -118.4424667, 118.5642853
20: -67.6727600, 52.7418060, -67.6440887, 52.5922318, -120.2649841, 120.3858948
21: -89.1764374, 53.7175903, -89.1145630, 53.5682449, -142.7446747, 142.8321533
22: -91.0156021, 55.2743340, -90.9624634, 55.0889969, -146.1045990, 146.2367859
23: -71.4732056, 55.8929939, -71.4384689, 55.7544060, -127.2275848, 127.3314667
24: -86.1371002, 51.4709854, -86.0971603, 51.3312187, -137.4683228, 137.5681458
25: -76.2993317, 59.1632042, -76.2622070, 58.9647980, -135.2641296, 135.4254150
26: -101.2590714, 82.4985733, -101.2093887, 82.2916870, -183.5507507, 183.7079468
27: -88.7836151, 60.0400696, -88.7408600, 59.8745079, -148.6581268, 148.7809296
28: -71.7865219, 61.1770782, -71.7613220, 60.9734993, -132.7600098, 132.9383850
29: -92.4262161, 48.8569946, -92.3621063, 48.6976128, -141.1238251, 141.2190857
30: -88.5565414, 65.5917969, -88.5138779, 65.4838257, -154.0403748, 154.1056671
31: -93.4090576, 55.3755188, -93.3576431, 55.1537857, -148.5628357, 148.7331543
32: -93.2362366, 59.3136864, -93.1936493, 59.2020493, -152.4382782, 152.5073395
33: -121.7482834, 78.9277878, -121.6835480, 78.7903137, -200.5385742, 200.6113281
34: -100.8021698, 58.1851883, -100.7586441, 57.9892120, -158.7913818, 158.9438171
35: -98.9569168, 60.8457680, -98.9133148, 60.6448479, -159.6017609, 159.7590790
36: -100.5130081, 63.0955429, -100.4753647, 62.8425446, -163.3555603, 163.5709076
37: -141.7996826, 62.0837936, -141.7215271, 62.0107727, -203.8104553, 203.8053284
38: -119.7875290, 78.9047165, -119.7336044, 78.6461716, -198.4337006, 198.6383209
39: -135.1265564, 75.6590652, -135.0393372, 75.5570602, -210.6836243, 210.6983643
40: -112.0707779, 61.3525848, -111.9968948, 61.2950516, -173.3657990, 173.3494720
41: -94.2461090, 64.7375641, -94.1943283, 64.6546326, -158.9007416, 158.9318848
42: -69.2583313, 60.0934944, -69.2225342, 60.0256081, -129.2839355, 129.3160248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5522923, upper bound: 81.4978437
time: 112.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5522923, upper bound: 81.5522922
time: 106.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 221.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 221.61
Output dim: 1, lower bound: -81.5522923, upper bound: 81.4697241
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 221.61
Output dim: 1, lower bound: -81.5522923, upper bound: 81.4950978
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 221.61
Output dim: 1, lower bound: -81.5522923, upper bound: 81.4978437
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 221.61
Output dim: 1, lower bound: -81.5522923, upper bound: 81.5522922

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -106.2661743, 78.3804855, -105.8078690, 78.3873901, -184.6535645, 184.1883545
1: -59.1131248, 65.5526505, -58.8174286, 65.5662155, -124.6793365, 124.3700790
2: -52.9285202, 63.1422920, -52.6754417, 63.1592026, -116.0877228, 115.8177338
3: -58.2832565, 72.5541687, -58.0493011, 72.5446472, -130.8278961, 130.6034698
4: -66.5831375, 73.7801208, -66.2465973, 73.7946014, -140.3777466, 140.0267029
5: -60.3673477, 70.3732300, -60.1110535, 70.3590240, -130.7263794, 130.4842834
6: -95.3185425, 62.4775009, -95.2966461, 62.2694244, -157.5879517, 157.7741241
7: -70.6144562, 67.3418121, -70.2858887, 67.3578186, -137.9722748, 137.6277008
8: -76.8987808, 89.8258820, -76.5523453, 89.8587036, -166.7574768, 166.3782043
9: -64.7459106, 71.0967484, -64.4503555, 71.1019897, -135.8479004, 135.5471039
10: -95.3637924, 90.6784210, -95.0657501, 90.6497650, -186.0135498, 185.7441711
11: -91.7091980, 56.0090752, -91.6364975, 55.8445206, -147.5537109, 147.6455688
12: -90.9152679, 72.6663361, -90.8821869, 72.4668884, -163.3821564, 163.5485229
13: -96.9668579, 95.5222092, -96.6507568, 95.4574356, -192.4242859, 192.1729736
14: -141.7035217, 80.0281677, -141.3348999, 80.0045013, -221.7080078, 221.3630676
15: -75.3416595, 67.1346817, -75.1625900, 67.1375427, -142.4792023, 142.2972717
16: -95.9948273, 69.3944855, -95.6969757, 69.3578568, -165.3526917, 165.0914612
17: -139.2025146, 76.1883087, -138.8513031, 76.1448364, -215.3473206, 215.0396118
18: -90.8066864, 75.3280411, -90.7727966, 75.1196747, -165.9263458, 166.1008301
19: -72.5951233, 45.5260468, -72.5868912, 45.2990646, -117.8941879, 118.1129303
20: -67.4830551, 52.4982529, -67.4777985, 52.2992287, -119.7822876, 119.9760437
21: -88.9039001, 53.4799690, -88.8771667, 53.2708015, -142.1746979, 142.3571320
22: -90.6974335, 54.9918594, -90.7016602, 54.7684517, -145.4658813, 145.6935120
23: -71.2581787, 55.6583366, -71.2292328, 55.4198532, -126.6780319, 126.8875732
24: -85.8830872, 51.2505150, -85.8695679, 51.0461578, -136.9292297, 137.1200867
25: -76.0342712, 58.8659782, -76.0544281, 58.6382370, -134.6725159, 134.9204102
26: -100.9909897, 82.1615753, -100.9204941, 81.8441010, -182.8350830, 183.0820618
27: -88.4839172, 59.7749977, -88.4738693, 59.5253792, -148.0092773, 148.2488708
28: -71.5341568, 60.8627243, -71.5323029, 60.5767365, -132.1109009, 132.3950195
29: -92.1099319, 48.6092529, -92.0962372, 48.3903275, -140.5002441, 140.7054901
30: -88.3374710, 65.3894119, -88.3142395, 65.2487335, -153.5862122, 153.7036438
31: -93.0846939, 55.0489807, -93.0973511, 54.7788773, -147.8635712, 148.1463318
32: -93.0214844, 59.1076927, -92.9835815, 58.9356689, -151.9571381, 152.0912781
33: -121.5072632, 78.6743546, -121.4299545, 78.4540710, -199.9613037, 200.1043091
34: -100.5883865, 57.8771019, -100.5559769, 57.6130676, -158.2014465, 158.4330750
35: -98.7160568, 60.5412750, -98.6885300, 60.2976837, -159.0137329, 159.2297974
36: -100.2531052, 62.7343445, -100.2493210, 62.4426117, -162.6957092, 162.9836578
37: -141.5215759, 61.9237900, -141.4250336, 61.7242508, -203.2457886, 203.3488159
38: -119.4745331, 78.5112000, -119.4477615, 78.2259064, -197.7003784, 197.9589386
39: -134.8548279, 75.4830780, -134.7672729, 75.3680573, -210.2228851, 210.2503510
40: -111.8539505, 61.2074356, -111.7834625, 61.0699272, -172.9238739, 172.9908752
41: -94.0494232, 64.5539322, -93.9746933, 64.3188629, -158.3682861, 158.5286255
42: -69.1376495, 59.9190559, -69.0841675, 59.7181473, -128.8558044, 129.0032196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4857151, upper bound: 81.4663621
time: 112.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5493605, upper bound: 81.4679865
time: 117.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -106.3987808, 78.3931427, -106.4542465, 78.6240845, -185.0228577, 184.8473816
1: -59.1949806, 65.5602417, -59.2154312, 65.7280579, -124.9230347, 124.7756729
2: -52.9987984, 63.1506500, -53.0126343, 63.3056221, -116.3044205, 116.1632843
3: -58.3475113, 72.5665359, -58.3594437, 72.7149658, -131.0624695, 130.9259796
4: -66.6753998, 73.7898865, -66.6950302, 73.9692841, -140.6446686, 140.4849091
5: -60.4376793, 70.3861237, -60.4524498, 70.5580597, -130.9957428, 130.8385620
6: -95.3362808, 62.5405197, -95.4533997, 62.5688286, -157.9051056, 157.9939117
7: -70.7053680, 67.3509064, -70.7241745, 67.5421600, -138.2475281, 138.0750732
8: -76.9942551, 89.8392029, -77.0180511, 90.0577545, -167.0520020, 166.8572540
9: -64.8290787, 71.1079102, -64.8600464, 71.2877350, -136.1168060, 135.9679565
10: -95.4483795, 90.6976547, -95.4883957, 90.8443146, -186.2926788, 186.1860199
11: -91.7407379, 56.0580711, -91.8390579, 56.0803261, -147.8210602, 147.8971252
12: -90.9319611, 72.7274628, -91.0341568, 72.7685776, -163.7005310, 163.7616272
13: -97.0572739, 95.5465546, -97.0909424, 95.7074814, -192.7647552, 192.6374969
14: -141.8136902, 80.0412674, -141.8775330, 80.1733170, -221.9869843, 221.9187927
15: -75.3955841, 67.1489944, -75.4334564, 67.2289200, -142.6245117, 142.5824432
16: -96.0858536, 69.4087601, -96.1443939, 69.5432434, -165.6290894, 165.5531616
17: -139.3055725, 76.2069168, -139.3591003, 76.2975464, -215.6031036, 215.5660095
18: -90.8292084, 75.3867340, -90.9596710, 75.4116974, -166.2409058, 166.3464050
19: -72.6127319, 45.5880623, -72.7660980, 45.5924149, -118.2051392, 118.3541565
20: -67.4992828, 52.5560455, -67.6222382, 52.5738487, -120.0731354, 120.1782837
21: -88.9300766, 53.5384483, -89.0850067, 53.5500679, -142.4801331, 142.6234589
22: -90.7209625, 55.0547943, -90.9288483, 55.0699539, -145.7909241, 145.9836426
23: -71.2746277, 55.7239151, -71.4125061, 55.7363930, -127.0110092, 127.1364212
24: -85.9065552, 51.3063354, -86.0650940, 51.3154030, -137.2219543, 137.3714294
25: -76.0526505, 58.9300232, -76.2309265, 58.9447784, -134.9974365, 135.1609497
26: -101.0170441, 82.2497406, -101.1787491, 82.2711945, -183.2882385, 183.4284973
27: -88.5054169, 59.8431320, -88.7063217, 59.8551521, -148.3605652, 148.5494385
28: -71.5487061, 60.9409828, -71.7310333, 60.9523010, -132.5009766, 132.6720123
29: -92.1360092, 48.6692810, -92.3299713, 48.6811218, -140.8171387, 140.9992371
30: -88.3624573, 65.4331512, -88.4849854, 65.4658127, -153.8282623, 153.9181213
31: -93.1065674, 55.1229706, -93.3222885, 55.1343460, -148.2409058, 148.4452515
32: -93.0414886, 59.1597519, -93.1691055, 59.1786537, -152.2201385, 152.3288574
33: -121.5373611, 78.7372742, -121.6566238, 78.7717896, -200.3091278, 200.3938904
34: -100.6080170, 57.9516373, -100.7335892, 57.9732628, -158.5812836, 158.6852112
35: -98.7382889, 60.6092262, -98.8839188, 60.6292610, -159.3675232, 159.4931335
36: -100.2701111, 62.8136902, -100.4445038, 62.8272743, -163.0973816, 163.2581940
37: -141.5543823, 61.9797630, -141.6915588, 61.9943657, -203.5487061, 203.6713257
38: -119.4981003, 78.5923767, -119.6974945, 78.6253815, -198.1234741, 198.2898560
39: -134.8947449, 75.5172653, -135.0091095, 75.5388031, -210.4335480, 210.5263519
40: -111.8822021, 61.2507858, -111.9760971, 61.2698746, -173.1520691, 173.2268829
41: -94.0716019, 64.6205139, -94.1735992, 64.6336136, -158.7052155, 158.7940979
42: -69.1534271, 59.9795609, -69.2100372, 60.0053978, -129.1588287, 129.1895905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1757

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4978438, upper bound: 81.4949912
time: 349.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4978438, upper bound: 81.4950979
time: 160.98 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -106.6844864, 78.6627502, -105.8150101, 78.4170837, -185.1015472, 184.4777527
1: -59.3983154, 65.7573242, -58.8206902, 65.5878983, -124.9862061, 124.5780106
2: -53.1583900, 63.3378830, -52.6776199, 63.1792717, -116.3376617, 116.0155029
3: -58.4838409, 72.7506409, -58.0513420, 72.5637589, -131.0476074, 130.8019867
4: -66.8558502, 74.0030060, -66.2497559, 73.8175964, -140.6734314, 140.2527618
5: -60.5603294, 70.5986481, -60.1128807, 70.3812256, -130.9415436, 130.7115326
6: -95.4946823, 62.6420059, -95.3115845, 62.2733803, -157.7680664, 157.9535828
7: -70.9110260, 67.5762024, -70.2884521, 67.3823242, -138.2933350, 137.8646545
8: -77.2432098, 90.1069336, -76.5563202, 89.8868790, -167.1300812, 166.6632538
9: -65.0078049, 71.3235550, -64.4548569, 71.1250000, -136.1328125, 135.7784119
10: -95.6604309, 90.8876190, -95.0712967, 90.6686554, -186.3290863, 185.9589081
11: -91.9171524, 56.1441460, -91.6481628, 55.8475571, -147.7647095, 147.7922974
12: -91.0741272, 72.8549271, -90.8952637, 72.4728241, -163.5469360, 163.7501831
13: -97.2029190, 95.7625885, -96.6550674, 95.4779358, -192.6808472, 192.4176331
14: -142.0043793, 80.2107544, -141.3420563, 80.0206757, -222.0250549, 221.5527954
15: -75.5387115, 67.2600708, -75.1679688, 67.1469727, -142.6856842, 142.4280396
16: -96.3291931, 69.5783920, -95.7044601, 69.3751755, -165.7043457, 165.2828522
17: -139.5337830, 76.3341446, -138.8572998, 76.1545868, -215.6883698, 215.1914368
18: -91.0176163, 75.5443420, -90.7897797, 75.1233521, -166.1409607, 166.3341064
19: -72.8168030, 45.7101517, -72.6064606, 45.3002853, -118.1170883, 118.3166122
20: -67.6566925, 52.6840363, -67.4935684, 52.3016129, -119.9583054, 120.1775970
21: -89.1505051, 53.6590118, -88.8970947, 53.2728996, -142.4234009, 142.5561066
22: -90.9922867, 55.2112350, -90.7282028, 54.7711220, -145.7633972, 145.9394379
23: -71.4569702, 55.8275375, -71.2469406, 55.4220924, -126.8790588, 127.0744705
24: -86.1138153, 51.4152145, -85.8901749, 51.0478516, -137.1616669, 137.3053894
25: -76.2811737, 59.0989075, -76.0775604, 58.6408081, -134.9219818, 135.1764526
26: -101.2334366, 82.4105759, -100.9416199, 81.8467560, -183.0802002, 183.3522034
27: -88.7624130, 59.9722786, -88.4997711, 59.5276642, -148.2900696, 148.4720459
28: -71.7722397, 61.0989876, -71.5558701, 60.5788727, -132.3511047, 132.6548615
29: -92.4001999, 48.7969627, -92.1207428, 48.3925476, -140.7927551, 140.9177094
30: -88.5317688, 65.5481720, -88.3299484, 65.2529297, -153.7846985, 153.8781128
31: -93.3874207, 55.3013878, -93.1251450, 54.7811661, -148.1685791, 148.4265289
32: -93.2162552, 59.2619133, -93.0001068, 58.9391212, -152.1553650, 152.2620087
33: -121.7184982, 78.8654480, -121.4449844, 78.4583588, -200.1768036, 200.3104248
34: -100.7827759, 58.1116943, -100.5726089, 57.6158066, -158.3985748, 158.6842957
35: -98.9350281, 60.7784882, -98.7073746, 60.3003578, -159.2353821, 159.4858704
36: -100.4962082, 63.0167160, -100.2723770, 62.4447021, -162.9409180, 163.2890930
37: -141.7669678, 62.0265045, -141.4423828, 61.7264481, -203.4934082, 203.4688873
38: -119.7642441, 78.8246460, -119.4745178, 78.2303925, -197.9946289, 198.2991486
39: -135.0873718, 75.6244125, -134.7810059, 75.3708801, -210.4582520, 210.4054260
40: -112.0425720, 61.3095970, -111.7945557, 61.0721474, -173.1147156, 173.1041412
41: -94.2239990, 64.6713715, -93.9876785, 64.3209686, -158.5449524, 158.6590576
42: -69.2425613, 60.0322227, -69.0907669, 59.7209930, -128.9635315, 129.1229858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4857151, upper bound: 81.4933599
time: 94.09 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5493605, upper bound: 81.4951722
time: 99.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -106.8154907, 78.6753693, -106.4606247, 78.6536102, -185.4690857, 185.1359863
1: -59.4805145, 65.7647858, -59.2184677, 65.7495270, -125.2300415, 124.9832535
2: -53.2288322, 63.3461914, -53.0144997, 63.3255196, -116.5543518, 116.3606796
3: -58.5482407, 72.7629242, -58.3611069, 72.7340088, -131.2822266, 131.1240234
4: -66.9480667, 74.0127106, -66.6975327, 73.9921570, -140.9402161, 140.7102356
5: -60.6308441, 70.6114120, -60.4537430, 70.5801239, -131.2109375, 131.0651550
6: -95.5125351, 62.7040367, -95.4682541, 62.5721283, -158.0846558, 158.1722870
7: -71.0024414, 67.5852814, -70.7264709, 67.5665741, -138.5690155, 138.3117371
8: -77.3389282, 90.1202164, -77.0215530, 90.0857162, -167.4246521, 167.1417542
9: -65.0898590, 71.3347473, -64.8640594, 71.3107910, -136.4006500, 136.1988068
10: -95.7453384, 90.9067078, -95.4933167, 90.8629456, -186.6082764, 186.4000244
11: -91.9487000, 56.1931839, -91.8513794, 56.0826073, -148.0313110, 148.0445557
12: -91.0908203, 72.9143219, -91.0470276, 72.7740250, -163.8648376, 163.9613495
13: -97.2929535, 95.7867126, -97.0940857, 95.7281723, -193.0210876, 192.8807831
14: -142.1135254, 80.2237549, -141.8835754, 80.1906204, -222.3041382, 222.1073303
15: -75.5925140, 67.2743225, -75.4382629, 67.2388458, -142.8313599, 142.7125854
16: -96.4193039, 69.5926208, -96.1516571, 69.5603943, -165.9796753, 165.7442780
17: -139.6358032, 76.3525314, -139.3642273, 76.3079834, -215.9437866, 215.7167664
18: -91.0398788, 75.6030121, -90.9763947, 75.4149017, -166.4547729, 166.5794067
19: -72.8342285, 45.7722473, -72.7856293, 45.5930939, -118.4273224, 118.5578766
20: -67.6727600, 52.7418060, -67.6377487, 52.5757256, -120.2484894, 120.3795547
21: -89.1764374, 53.7175903, -89.1047745, 53.5515671, -142.7279968, 142.8223572
22: -91.0156021, 55.2743340, -90.9551086, 55.0723343, -146.0879364, 146.2294464
23: -71.4732056, 55.8929939, -71.4301987, 55.7379990, -127.2112045, 127.3231964
24: -86.1371002, 51.4709854, -86.0856247, 51.3165703, -137.4536438, 137.5566101
25: -76.2993317, 59.1632042, -76.2539291, 58.9469604, -135.2462921, 135.4171295
26: -101.2590714, 82.4985733, -101.1994095, 82.2733231, -183.5323792, 183.6979828
27: -88.7836151, 60.0400696, -88.7321014, 59.8565826, -148.6401978, 148.7721710
28: -71.7865219, 61.1770782, -71.7544403, 60.9537506, -132.7402649, 132.9315033
29: -92.4262161, 48.8569946, -92.3544083, 48.6829758, -141.1091766, 141.2113953
30: -88.5565414, 65.5917969, -88.5006409, 65.4693451, -154.0258789, 154.0924377
31: -93.4090576, 55.3755188, -93.3500824, 55.1361046, -148.5451660, 148.7256012
32: -93.2362366, 59.3136864, -93.1853714, 59.1815987, -152.4178314, 152.4990540
33: -121.7482834, 78.9277878, -121.6717300, 78.7757797, -200.5240479, 200.5995178
34: -100.8021698, 58.1851883, -100.7489700, 57.9757919, -158.7779541, 158.9341583
35: -98.9569168, 60.8457680, -98.9023819, 60.6316147, -159.5885315, 159.7481537
36: -100.5130081, 63.0955429, -100.4671860, 62.8290749, -163.3420715, 163.5627289
37: -141.7996826, 62.0837936, -141.7080994, 61.9963074, -203.7959595, 203.7918854
38: -119.7875290, 78.9047165, -119.7223129, 78.6294708, -198.4169922, 198.6270294
39: -135.1265564, 75.6590652, -135.0227356, 75.5414124, -210.6679688, 210.6817932
40: -112.0707779, 61.3525848, -111.9873886, 61.2713699, -173.3421478, 173.3399658
41: -94.2461090, 64.7375641, -94.1859589, 64.6349640, -158.8810730, 158.9235229
42: -69.2583313, 60.0934944, -69.2165985, 60.0072212, -129.2655487, 129.3100891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1757

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4978438, upper bound: 81.5522922
time: 223.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4978438, upper bound: 81.5522922
time: 103.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 329.25 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.4857151, upper bound: 81.4663621
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.5493605, upper bound: 81.4679865
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.4978438, upper bound: 81.4949912
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.4978438, upper bound: 81.4950979
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.4857151, upper bound: 81.4933599
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.5493605, upper bound: 81.4951722
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.4978438, upper bound: 81.5522922
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 329.25
Output dim: 1, lower bound: -81.4978438, upper bound: 81.5522922

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -105.6654358, 78.1906586, -105.6900558, 78.3780518, -184.0434875, 183.8807068
1: -58.7284813, 65.4153824, -58.7409554, 65.5606308, -124.2891083, 124.1563416
2: -52.6044159, 63.0136681, -52.6106758, 63.1525269, -115.7569427, 115.6243362
3: -57.9785500, 72.4119568, -57.9883766, 72.5343399, -130.5128937, 130.4003296
4: -66.1782532, 73.6366043, -66.1663513, 73.7867584, -139.9650116, 139.8029480
5: -60.0361252, 70.2166672, -60.0448875, 70.3490143, -130.3851318, 130.2615509
6: -95.1784668, 62.1613197, -95.2838821, 62.2075043, -157.3859711, 157.4452057
7: -70.1755905, 67.1770782, -70.1975708, 67.3496857, -137.5252686, 137.3746490
8: -76.4208603, 89.6361847, -76.4570999, 89.8472290, -166.2680969, 166.0932770
9: -64.3652573, 70.9363022, -64.3752747, 71.0924149, -135.4576416, 135.3115845
10: -94.9486313, 90.5089188, -94.9846573, 90.6335449, -185.5821838, 185.4935608
11: -91.5711517, 55.8258324, -91.6127472, 55.8095589, -147.3807068, 147.4385834
12: -90.7644424, 72.2962036, -90.8682709, 72.3958130, -163.1602478, 163.1644745
13: -96.6171265, 95.3705902, -96.5830765, 95.4379120, -192.0550385, 191.9536438
14: -141.2231750, 79.8946838, -141.2423859, 79.9938202, -221.2169647, 221.1370697
15: -75.0753250, 67.0742874, -75.1126556, 67.1286621, -142.2039795, 142.1869507
16: -95.5803070, 69.2415619, -95.6156921, 69.3452911, -164.9255829, 164.8572540
17: -138.7706604, 76.0781326, -138.7680969, 76.1291656, -214.8997650, 214.8462219
18: -90.6366501, 75.0205841, -90.7538452, 75.0590973, -165.6957397, 165.7744293
19: -72.4611511, 45.2758713, -72.5726242, 45.2484970, -117.7096405, 117.8484879
20: -67.3669357, 52.2494164, -67.4652863, 52.2501068, -119.6170425, 119.7147064
21: -88.7396851, 53.2401237, -88.8558884, 53.2234306, -141.9631042, 142.0960083
22: -90.5117569, 54.7090378, -90.6814880, 54.7117424, -145.2234955, 145.3905334
23: -71.1353378, 55.4127731, -71.2163773, 55.3715096, -126.5068512, 126.6291504
24: -85.7456284, 51.0330887, -85.8520737, 51.0032425, -136.7488708, 136.8851624
25: -75.8816223, 58.5861130, -76.0390930, 58.5828056, -134.4644165, 134.6251831
26: -100.7819977, 81.7619781, -100.9008865, 81.7642517, -182.5462494, 182.6628723
27: -88.3241196, 59.5080757, -88.4572983, 59.4731712, -147.7972870, 147.9653625
28: -71.3941193, 60.5434341, -71.5212021, 60.5135078, -131.9076233, 132.0646362
29: -91.9314423, 48.3678284, -92.0738754, 48.3422813, -140.2737122, 140.4416962
30: -88.2202225, 65.2119064, -88.2953644, 65.2154617, -153.4356842, 153.5072632
31: -92.9011993, 54.7146530, -93.0789642, 54.7121048, -147.6133118, 147.7936096
32: -92.8755569, 58.8303528, -92.9692535, 58.8815079, -151.7570648, 151.7996063
33: -121.3054657, 78.3197556, -121.4069061, 78.3845673, -199.6900177, 199.7266541
34: -100.4256363, 57.5081902, -100.5404816, 57.5392151, -157.9648438, 158.0486755
35: -98.5320511, 60.1721420, -98.6711731, 60.2237663, -158.7558136, 158.8433228
36: -100.0658875, 62.3055534, -100.2357559, 62.3563843, -162.4222717, 162.5413055
37: -141.2830353, 61.6105690, -141.3990936, 61.6617737, -202.9448090, 203.0096436
38: -119.2287674, 78.0427856, -119.4279938, 78.1327515, -197.3615112, 197.4707794
39: -134.6460876, 75.2295380, -134.7387085, 75.3175659, -209.9636230, 209.9682465
40: -111.6805115, 60.9416733, -111.7619171, 61.0174713, -172.6979675, 172.7035828
41: -93.8958206, 64.2693100, -93.9587250, 64.2627335, -158.1585388, 158.2280273
42: -69.0388489, 59.6563301, -69.0727310, 59.6667328, -128.7055664, 128.7290649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 733

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4565583, upper bound: 81.4663621
time: 146.25 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4565583, upper bound: 81.4663621
time: 105.24 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -106.2425537, 78.3776398, -105.8078690, 78.3873901, -184.6299438, 184.1855164
1: -59.0977135, 65.5508957, -58.8174286, 65.5662155, -124.6639252, 124.3683243
2: -52.9152870, 63.1400948, -52.6754417, 63.1592026, -116.0744934, 115.8155365
3: -58.2706985, 72.5513153, -58.0493011, 72.5446472, -130.8153381, 130.6006165
4: -66.5662079, 73.7774658, -66.2465973, 73.7946014, -140.3608093, 140.0240479
5: -60.3535652, 70.3700867, -60.1110535, 70.3590240, -130.7125854, 130.4811249
6: -95.3151703, 62.4636269, -95.2966461, 62.2694244, -157.5845795, 157.7602692
7: -70.5966797, 67.3391266, -70.2858887, 67.3578186, -137.9544983, 137.6250153
8: -76.8790665, 89.8227081, -76.5523453, 89.8587036, -166.7377625, 166.3750305
9: -64.7307892, 71.0942230, -64.4503555, 71.1019897, -135.8327789, 135.5445862
10: -95.3470383, 90.6736603, -95.0657501, 90.6497650, -185.9967957, 185.7394104
11: -91.7023849, 55.9996872, -91.6364975, 55.8445206, -147.5468750, 147.6361847
12: -90.9113922, 72.6507263, -90.8821869, 72.4668884, -163.3782501, 163.5328979
13: -96.9513397, 95.5172043, -96.6507568, 95.4574356, -192.4087524, 192.1679688
14: -141.6826477, 80.0254669, -141.3348999, 80.0045013, -221.6871338, 221.3603668
15: -75.3298187, 67.1314850, -75.1625900, 67.1375427, -142.4673615, 142.2940674
16: -95.9769669, 69.3911819, -95.6969757, 69.3578568, -165.3348236, 165.0881653
17: -139.1850433, 76.1842194, -138.8513031, 76.1448364, -215.3298645, 215.0355225
18: -90.8012161, 75.3145828, -90.7727966, 75.1196747, -165.9208832, 166.0873718
19: -72.5910187, 45.5153198, -72.5868912, 45.2990646, -117.8900833, 118.1022110
20: -67.4791489, 52.4865799, -67.4777985, 52.2992287, -119.7783585, 119.9643784
21: -88.8979111, 53.4690552, -88.8771667, 53.2708015, -142.1687164, 142.3462219
22: -90.6920853, 54.9798660, -90.7016602, 54.7684517, -145.4605408, 145.6815186
23: -71.2540817, 55.6474686, -71.2292328, 55.4198532, -126.6739349, 126.8767014
24: -85.8773422, 51.2406731, -85.8695679, 51.0461578, -136.9234924, 137.1102295
25: -76.0293274, 58.8536644, -76.0544281, 58.6382370, -134.6675720, 134.9080963
26: -100.9849091, 82.1446991, -100.9204941, 81.8441010, -182.8290100, 183.0651855
27: -88.4788208, 59.7636604, -88.4738693, 59.5253792, -148.0041962, 148.2375183
28: -71.5300751, 60.8495598, -71.5323029, 60.5767365, -132.1068115, 132.3818665
29: -92.1042633, 48.5994339, -92.0962372, 48.3903275, -140.4945984, 140.6956635
30: -88.3303680, 65.3808289, -88.3142395, 65.2487335, -153.5791016, 153.6950684
31: -93.0801086, 55.0348053, -93.0973511, 54.7788773, -147.8589783, 148.1321411
32: -93.0173416, 59.0948563, -92.9835815, 58.9356689, -151.9530029, 152.0784302
33: -121.5014648, 78.6595383, -121.4299545, 78.4540710, -199.9555359, 200.0894775
34: -100.5836945, 57.8622932, -100.5559769, 57.6130676, -158.1967621, 158.4182739
35: -98.7107849, 60.5269928, -98.6885300, 60.2976837, -159.0084534, 159.2155151
36: -100.2491302, 62.7176323, -100.2493210, 62.4426117, -162.6917114, 162.9669342
37: -141.5141296, 61.9107056, -141.4250336, 61.7242508, -203.2383728, 203.3357391
38: -119.4693298, 78.4921722, -119.4477615, 78.2259064, -197.6952209, 197.9398956
39: -134.8470764, 75.4721451, -134.7672729, 75.3680573, -210.2151337, 210.2394104
40: -111.8483582, 61.1984787, -111.7834625, 61.0699272, -172.9182892, 172.9819336
41: -94.0453186, 64.5417252, -93.9746933, 64.3188629, -158.3641663, 158.5164185
42: -69.1344757, 59.9064980, -69.0841675, 59.7181473, -128.8526306, 128.9906616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 733

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5237248, upper bound: 81.4679865
time: 119.31 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5237248, upper bound: 81.4679865
time: 2104.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2225.82 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2225.82
Output dim: 1, lower bound: -81.4565583, upper bound: 81.4663621
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2225.82
Output dim: 1, lower bound: -81.4565583, upper bound: 81.4663621
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2225.82
Output dim: 1, lower bound: -81.5237248, upper bound: 81.4679865
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2225.82
Output dim: 1, lower bound: -81.5237248, upper bound: 81.4679865
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2225.82
Output dim: 1, lower bound: -81.4978438, upper bound: 81.4949912
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2225.82
Output dim: 1, lower bound: -81.4978438, upper bound: 81.4950979
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2225.82
Output dim: 1, lower bound: -81.4857151, upper bound: 81.4933599
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2225.82
Output dim: 1, lower bound: -81.5493605, upper bound: 81.4951722
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2225.82
Output dim: 1, lower bound: -81.4978438, upper bound: 81.5522922
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2225.82
Output dim: 1, lower bound: -81.4978438, upper bound: 81.5522922
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=124.9945068359375
rel_dist={1: [-81.5568719651244, 81.5568719651244]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2392194, upper bound: 80.1916233
time: 116.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2392194, upper bound: 80.2392193
time: 163.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 279.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 279.50
Output dim: 1, lower bound: -80.2392194, upper bound: 80.1916233
IS_A2, status: Status.UNKNOWN, split count: 1, time: 279.50
Output dim: 1, lower bound: -80.2392194, upper bound: 80.2392193

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -106.3987808, 78.3931427, -106.4687729, 78.5949326, -184.9937134, 184.8619080
1: -59.1949806, 65.5602417, -59.2282562, 65.7066040, -124.9015808, 124.7884979
2: -52.9987984, 63.1506500, -53.0255661, 63.2863998, -116.2852020, 116.1762161
3: -58.3475113, 72.5665359, -58.3711472, 72.6975327, -131.0450439, 130.9376678
4: -66.6753998, 73.7898865, -66.7104721, 73.9473114, -140.6226807, 140.5003662
5: -60.4376793, 70.3861237, -60.4631996, 70.5373840, -130.9750671, 130.8493195
6: -95.3362808, 62.5405197, -95.4420395, 62.5836868, -157.9199677, 157.9825592
7: -70.7053680, 67.3509064, -70.7375183, 67.5175018, -138.2228699, 138.0884247
8: -76.9942551, 89.8392029, -77.0337982, 90.0315857, -167.0258484, 166.8729858
9: -64.8290787, 71.1079102, -64.8681183, 71.2662964, -136.0953674, 135.9760284
10: -95.4483795, 90.6976547, -95.4968948, 90.8301086, -186.2784882, 186.1945343
11: -91.7407379, 56.0580711, -91.8353500, 56.0891838, -147.8299255, 147.8934021
12: -90.9319611, 72.7274628, -91.0250702, 72.7784271, -163.7103577, 163.7525330
13: -97.0572739, 95.5465546, -97.1055222, 95.6917419, -192.7490234, 192.6520691
14: -141.8136902, 80.0412674, -141.8887634, 80.1583557, -221.9720459, 221.9300232
15: -75.3955841, 67.1489944, -75.4433060, 67.2229156, -142.6184998, 142.5922852
16: -96.0858536, 69.4087601, -96.1527252, 69.5289688, -165.6148071, 165.5614929
17: -139.3055725, 76.2069168, -139.3710785, 76.2893066, -215.5948792, 215.5780029
18: -90.8292084, 75.3867340, -90.9490433, 75.4207306, -166.2499390, 166.3357849
19: -72.6127319, 45.5880623, -72.7492065, 45.6046410, -118.2173615, 118.3372650
20: -67.4992828, 52.5560455, -67.6097488, 52.5852661, -120.0845337, 120.1657867
21: -88.9300766, 53.5384483, -89.0707703, 53.5625648, -142.4926453, 142.6092072
22: -90.7209625, 55.0547943, -90.9049530, 55.0818558, -145.8028259, 145.9597473
23: -71.2746277, 55.7239151, -71.3994446, 55.7484818, -127.0231094, 127.1233597
24: -85.9065552, 51.3063354, -86.0517654, 51.3265076, -137.2330322, 137.3580933
25: -76.0526505, 58.9300232, -76.2121124, 58.9577789, -135.0104218, 135.1421356
26: -101.0170441, 82.2497406, -101.1636581, 82.2836990, -183.3007507, 183.4133911
27: -88.5054169, 59.8431320, -88.6845551, 59.8685989, -148.3740234, 148.5276794
28: -71.5487061, 60.9409828, -71.7104340, 60.9673843, -132.5160828, 132.6514130
29: -92.1360092, 48.6692810, -92.3084488, 48.6918526, -140.8278656, 140.9777222
30: -88.3624573, 65.4331512, -88.4782562, 65.4733582, -153.8358154, 153.9114075
31: -93.1065674, 55.1229706, -93.2973404, 55.1477280, -148.2542877, 148.4203186
32: -93.0414886, 59.1597519, -93.1574326, 59.1932716, -152.2347565, 152.3171844
33: -121.5373611, 78.7372742, -121.6492538, 78.7790833, -200.3163757, 200.3865204
34: -100.6080170, 57.9516373, -100.7234802, 57.9815598, -158.5895538, 158.6751099
35: -98.7382889, 60.6092262, -98.8720093, 60.6375656, -159.3758392, 159.4812317
36: -100.2701111, 62.8136902, -100.4258575, 62.8367500, -163.1068573, 163.2395477
37: -141.5543823, 61.9797630, -141.6828613, 62.0045776, -203.5589294, 203.6626282
38: -119.4981003, 78.5923767, -119.6781998, 78.6347580, -198.1328278, 198.2705688
39: -134.8947449, 75.5172653, -135.0065155, 75.5489807, -210.4437256, 210.5237732
40: -111.8822021, 61.2507858, -111.9703751, 61.2872849, -173.1694946, 173.2211609
41: -94.0716019, 64.6205139, -94.1658401, 64.6484375, -158.7200165, 158.7863464
42: -69.1534271, 59.9795609, -69.2067184, 60.0170975, -129.1705322, 129.1862793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2379531, upper bound: 80.1684823
time: 93.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2379531, upper bound: 80.1878689
time: 100.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -106.8154907, 78.6753693, -106.4837189, 78.6562347, -185.4717102, 185.1590881
1: -59.4805145, 65.7647858, -59.2353363, 65.7511520, -125.2316666, 125.0001144
2: -53.2288322, 63.3461914, -53.0303268, 63.3277054, -116.5565186, 116.3765030
3: -58.5482407, 72.7629242, -58.3753242, 72.7371368, -131.2853699, 131.1382446
4: -66.9480667, 74.0127106, -66.7168274, 73.9948959, -140.9429474, 140.7295380
5: -60.6308441, 70.6114120, -60.4669800, 70.5832062, -131.2140350, 131.0783844
6: -95.5125351, 62.7040367, -95.4732895, 62.5918846, -158.1044159, 158.1773224
7: -71.0024414, 67.5852814, -70.7433167, 67.5681458, -138.5705719, 138.3285980
8: -77.3389282, 90.1202164, -77.0421295, 90.0897827, -167.4287109, 167.1623535
9: -65.0898590, 71.3347473, -64.8772354, 71.3142242, -136.4040833, 136.2119751
10: -95.7453384, 90.9067078, -95.5081024, 90.8692169, -186.6145630, 186.4147949
11: -91.9487000, 56.1931839, -91.8616791, 56.0949631, -148.0436554, 148.0548401
12: -91.0908203, 72.9143219, -91.0522461, 72.7907715, -163.8815613, 163.9665680
13: -97.2929535, 95.7867126, -97.1138916, 95.7349625, -193.0279236, 192.9005890
14: -142.1135254, 80.2237549, -141.9035950, 80.1938782, -222.3073730, 222.1273499
15: -75.5925140, 67.2743225, -75.4543457, 67.2441254, -142.8366241, 142.7286682
16: -96.4193039, 69.5926208, -96.1689377, 69.5648422, -165.9841309, 165.7615662
17: -139.6358032, 76.3525314, -139.3836975, 76.3116379, -215.9474487, 215.7362366
18: -91.0398788, 75.6030121, -90.9841919, 75.4281616, -166.4680481, 166.5871887
19: -72.8342285, 45.7722473, -72.7899857, 45.6068497, -118.4410782, 118.5622253
20: -67.6727600, 52.7418060, -67.6422729, 52.5902405, -120.2630005, 120.3840790
21: -89.1764374, 53.7175903, -89.1122131, 53.5665703, -142.7429962, 142.8298035
22: -91.0156021, 55.2743340, -90.9598694, 55.0875244, -146.1031189, 146.2341919
23: -71.4732056, 55.8929939, -71.4364624, 55.7527618, -127.2259674, 127.3294525
24: -86.1371002, 51.4709854, -86.0947800, 51.3297462, -137.4668427, 137.5657654
25: -76.2993317, 59.1632042, -76.2600403, 58.9631500, -135.2624817, 135.4232483
26: -101.2590714, 82.4985733, -101.2069550, 82.2892990, -183.5483398, 183.7055206
27: -88.7836151, 60.0400696, -88.7383347, 59.8726768, -148.6562805, 148.7784119
28: -71.7865219, 61.1770782, -71.7591858, 60.9715500, -132.7580719, 132.9362488
29: -92.4262161, 48.8569946, -92.3595276, 48.6963272, -141.1225433, 141.2165070
30: -88.5565414, 65.5917969, -88.5115051, 65.4817505, -154.0382996, 154.1033020
31: -93.4090576, 55.3755188, -93.3551483, 55.1522141, -148.5612640, 148.7306671
32: -93.2362366, 59.3136864, -93.1916656, 59.2002716, -152.4365082, 152.5053406
33: -121.7482834, 78.9277878, -121.6811218, 78.7883682, -200.5366364, 200.6089172
34: -100.8021698, 58.1851883, -100.7560959, 57.9875870, -158.7897644, 158.9412689
35: -98.9569168, 60.8457680, -98.9109268, 60.6433105, -159.6002197, 159.7566986
36: -100.5130081, 63.0955429, -100.4731903, 62.8411713, -163.3541870, 163.5687256
37: -141.7996826, 62.0837936, -141.7181702, 62.0093613, -203.8090210, 203.8019562
38: -119.7875290, 78.9047165, -119.7307358, 78.6442108, -198.4317169, 198.6354523
39: -135.1265564, 75.6590652, -135.0359955, 75.5553207, -210.6818848, 210.6950684
40: -112.0707779, 61.3525848, -111.9945984, 61.2919426, -173.3627167, 173.3471680
41: -94.2461090, 64.7375641, -94.1921158, 64.6524582, -158.8985596, 158.9296875
42: -69.2583313, 60.0934944, -69.2209396, 60.0227165, -129.2810364, 129.3144379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2379531, upper bound: 80.1928271
time: 158.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2379531, upper bound: 80.2379530
time: 114.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 275.24 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 275.24
Output dim: 1, lower bound: -80.2379531, upper bound: 80.1684823
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 275.24
Output dim: 1, lower bound: -80.2379531, upper bound: 80.1878689
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 275.24
Output dim: 1, lower bound: -80.2379531, upper bound: 80.1928271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 275.24
Output dim: 1, lower bound: -80.2379531, upper bound: 80.2379530

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -106.2036667, 78.3744812, -105.7957764, 78.3533783, -184.5570374, 184.1702576
1: -59.0743103, 65.5490723, -58.8117180, 65.5414429, -124.6157532, 124.3607941
2: -52.8952217, 63.1383362, -52.6707954, 63.1362228, -116.0314484, 115.8091125
3: -58.2528076, 72.5483398, -58.0451965, 72.5224838, -130.7752991, 130.5935364
4: -66.5394592, 73.7754898, -66.2404633, 73.7680283, -140.3074799, 140.0159607
5: -60.3340111, 70.3671188, -60.1065788, 70.3334198, -130.6674347, 130.4736938
6: -95.3101578, 62.4477425, -95.2785110, 62.2618217, -157.5719757, 157.7262573
7: -70.5713730, 67.3374939, -70.2803192, 67.3297501, -137.9011230, 137.6178131
8: -76.8535461, 89.8195724, -76.5454407, 89.8262939, -166.6798401, 166.3650208
9: -64.7071075, 71.0914688, -64.4435883, 71.0751648, -135.7822723, 135.5350647
10: -95.3236923, 90.6692810, -95.0572739, 90.6272125, -185.9508972, 185.7265625
11: -91.6942978, 55.9859123, -91.6205063, 55.8390121, -147.5333099, 147.6064148
12: -90.9073639, 72.6374969, -90.8661194, 72.4578094, -163.3651581, 163.5036163
13: -96.9240875, 95.5106659, -96.6422729, 95.4327087, -192.3567810, 192.1529388
14: -141.6519318, 80.0219650, -141.3218689, 79.9847565, -221.6366882, 221.3438416
15: -75.3161316, 67.1279068, -75.1542130, 67.1249390, -142.4410706, 142.2821198
16: -95.9523544, 69.3877640, -95.6855087, 69.3374329, -165.2897949, 165.0732727
17: -139.1538086, 76.1794739, -138.8399658, 76.1311340, -215.2849121, 215.0194397
18: -90.7960739, 75.3002472, -90.7522888, 75.1137314, -165.9097900, 166.0525360
19: -72.5867920, 45.4966125, -72.5636749, 45.2960892, -117.8828812, 118.0602798
20: -67.4754028, 52.4708710, -67.4589539, 52.2940445, -119.7694473, 119.9298172
21: -88.8915024, 53.4522438, -88.8531952, 53.2664871, -142.1579895, 142.3054352
22: -90.6863098, 54.9620171, -90.6705475, 54.7636719, -145.4499817, 145.6325684
23: -71.2504120, 55.6272583, -71.2079239, 55.4155006, -126.6659088, 126.8351822
24: -85.8719711, 51.2240257, -85.8447952, 51.0425339, -136.9145050, 137.0688171
25: -76.0256042, 58.8356094, -76.0273972, 58.6333313, -134.6589203, 134.8630066
26: -100.9786606, 82.1198349, -100.8952942, 81.8381500, -182.8168030, 183.0151367
27: -88.4737625, 59.7426872, -88.4433975, 59.5208130, -147.9945679, 148.1860809
28: -71.5272903, 60.8256149, -71.5048218, 60.5720444, -132.0993195, 132.3304138
29: -92.0976562, 48.5808144, -92.0671005, 48.3863525, -140.4839935, 140.6479187
30: -88.3256760, 65.3687134, -88.2944717, 65.2416687, -153.5673523, 153.6631775
31: -93.0743332, 55.0139084, -93.0649567, 54.7744598, -147.8487854, 148.0788574
32: -93.0120010, 59.0833397, -92.9636917, 58.9297447, -151.9417419, 152.0470276
33: -121.4929962, 78.6445618, -121.4107285, 78.4467621, -199.9397278, 200.0552826
34: -100.5791397, 57.8417587, -100.5357666, 57.6079102, -158.1870422, 158.3775330
35: -98.7055740, 60.5096130, -98.6657410, 60.2927666, -158.9983215, 159.1753540
36: -100.2450485, 62.6972084, -100.2225266, 62.4385757, -162.6836243, 162.9197388
37: -141.5059814, 61.8972092, -141.4027100, 61.7199402, -203.2259216, 203.2998962
38: -119.4634628, 78.4727020, -119.4166031, 78.2186127, -197.6820374, 197.8892975
39: -134.8360443, 75.4668808, -134.7481079, 75.3625488, -210.1985931, 210.2149658
40: -111.8406296, 61.1869125, -111.7685394, 61.0635071, -172.9041138, 172.9554443
41: -94.0389099, 64.5223694, -93.9584122, 64.3138962, -158.3528137, 158.4807739
42: -69.1302032, 59.8905525, -69.0750580, 59.7114868, -128.8416901, 128.9656067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1627925
time: 100.94 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1670023
time: 157.24 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -106.3954163, 78.3925476, -106.4423370, 78.5900421, -184.9854584, 184.8348846
1: -59.1925888, 65.5598221, -59.2098007, 65.7033691, -124.8959579, 124.7696228
2: -52.9966202, 63.1502037, -53.0080910, 63.2827263, -116.2793350, 116.1582947
3: -58.3455200, 72.5658722, -58.3554459, 72.6927948, -131.0383148, 130.9213257
4: -66.6727295, 73.7893143, -66.6890945, 73.9427490, -140.6154785, 140.4784088
5: -60.4354630, 70.3854980, -60.4481468, 70.5325165, -130.9679871, 130.8336334
6: -95.3354034, 62.5377350, -95.4351883, 62.5614281, -157.8968353, 157.9729309
7: -70.7027283, 67.3504486, -70.7186890, 67.5141068, -138.2168274, 138.0691223
8: -76.9914246, 89.8383865, -77.0112762, 90.0253754, -167.0167999, 166.8496704
9: -64.8272095, 71.1072540, -64.8534088, 71.2609177, -136.0881348, 135.9606628
10: -95.4462662, 90.6965942, -95.4801254, 90.8218536, -186.2681274, 186.1767273
11: -91.7391434, 56.0562515, -91.8227158, 56.0750732, -147.8142090, 147.8789673
12: -90.9310379, 72.7251282, -91.0181046, 72.7597809, -163.6908264, 163.7432251
13: -97.0540009, 95.5454102, -97.0827789, 95.6826782, -192.7366638, 192.6281738
14: -141.8106232, 80.0406342, -141.8647156, 80.1534882, -221.9641113, 221.9053345
15: -75.3932037, 67.1481018, -75.4253235, 67.2158356, -142.6090393, 142.5734253
16: -96.0833511, 69.4079590, -96.1329651, 69.5228958, -165.6062469, 165.5409241
17: -139.3025513, 76.2061920, -139.3479919, 76.2834473, -215.5859985, 215.5541687
18: -90.8279572, 75.3846741, -90.9392624, 75.4059448, -166.2338867, 166.3239441
19: -72.6119385, 45.5861740, -72.7427979, 45.5896149, -118.2015381, 118.3289719
20: -67.4984894, 52.5538521, -67.6034698, 52.5688858, -120.0673676, 120.1573181
21: -88.9288635, 53.5363846, -89.0610275, 53.5459671, -142.4748230, 142.5974121
22: -90.7199860, 55.0527534, -90.8976898, 55.0652695, -145.7852478, 145.9504395
23: -71.2735901, 55.7219009, -71.3912201, 55.7322273, -127.0058136, 127.1131058
24: -85.9051208, 51.3045349, -86.0403137, 51.3119698, -137.2170868, 137.3448486
25: -76.0516129, 58.9278145, -76.2038956, 58.9400330, -134.9916382, 135.1317139
26: -101.0157776, 82.2472763, -101.1537781, 82.2654877, -183.2812653, 183.4010315
27: -88.5043182, 59.8409271, -88.6758728, 59.8508453, -148.3551636, 148.5167999
28: -71.5478516, 60.9385452, -71.7036133, 60.9478073, -132.4956665, 132.6421509
29: -92.1350098, 48.6674805, -92.3007812, 48.6772614, -140.8122711, 140.9682617
30: -88.3608551, 65.4313660, -88.4651489, 65.4589996, -153.8198547, 153.8965149
31: -93.1056137, 55.1207314, -93.2898407, 55.1301231, -148.2357330, 148.4105682
32: -93.0404510, 59.1572266, -93.1492386, 59.1729164, -152.2133636, 152.3064575
33: -121.5359344, 78.7354431, -121.6375656, 78.7646942, -200.3006287, 200.3730164
34: -100.6068115, 57.9499245, -100.7138901, 57.9681587, -158.5749512, 158.6638031
35: -98.7369461, 60.6075134, -98.8612442, 60.6244621, -159.3613892, 159.4687500
36: -100.2691116, 62.8119774, -100.4178162, 62.8233948, -163.0924988, 163.2297974
37: -141.5527039, 61.9774551, -141.6695709, 61.9901962, -203.5428925, 203.6470337
38: -119.4966812, 78.5903091, -119.6670990, 78.6182175, -198.1148987, 198.2574158
39: -134.8927155, 75.5153046, -134.9900818, 75.5333862, -210.4260864, 210.5053864
40: -111.8810120, 61.2478981, -111.9609299, 61.2636604, -173.1446533, 173.2088318
41: -94.0705490, 64.6180878, -94.1575317, 64.6289062, -158.6994476, 158.7756195
42: -69.1526947, 59.9770393, -69.2008591, 59.9990845, -129.1517792, 129.1778870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1826401
time: 91.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2355561, upper bound: 80.1863503
time: 1180.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -106.6224060, 78.6567841, -105.8118210, 78.4149017, -185.0373077, 184.4685974
1: -59.3592911, 65.7537918, -58.8191032, 65.5863800, -124.9456711, 124.5728912
2: -53.1249771, 63.3339806, -52.6760368, 63.1778564, -116.3028336, 116.0100174
3: -58.4533195, 72.7448273, -58.0499649, 72.5621490, -131.0154724, 130.7947998
4: -66.8121490, 73.9983978, -66.2478180, 73.8157654, -140.6279144, 140.2462158
5: -60.5269012, 70.5926208, -60.1111107, 70.3794632, -130.9063721, 130.7037354
6: -95.4862289, 62.6129837, -95.3097916, 62.2710876, -157.7573242, 157.9227600
7: -70.8676682, 67.5718994, -70.2864914, 67.3805695, -138.2482300, 137.8583984
8: -77.1978302, 90.1006927, -76.5544128, 89.8847504, -167.0825806, 166.6550903
9: -64.9688950, 71.3182297, -64.4534454, 71.1230774, -136.0919495, 135.7716675
10: -95.6202164, 90.8785858, -95.0694046, 90.6667023, -186.2869263, 185.9479675
11: -91.9021835, 56.1210251, -91.6457977, 55.8459625, -147.7481384, 147.7668152
12: -91.0662613, 72.8267746, -90.8935242, 72.4708710, -163.5371094, 163.7202911
13: -97.1603546, 95.7511444, -96.6523743, 95.4756851, -192.6360168, 192.4035187
14: -141.9527435, 80.2046356, -141.3382416, 80.0185623, -221.9713135, 221.5428772
15: -75.5132294, 67.2533264, -75.1661377, 67.1451645, -142.6583862, 142.4194489
16: -96.2864151, 69.5716400, -95.7019577, 69.3735504, -165.6599731, 165.2735901
17: -139.4854889, 76.3254471, -138.8538208, 76.1520386, -215.6375122, 215.1792450
18: -91.0071030, 75.5165863, -90.7878342, 75.1218872, -166.1289978, 166.3044128
19: -72.8085403, 45.6807518, -72.6045227, 45.2991104, -118.1076508, 118.2852631
20: -67.6491089, 52.6567078, -67.4919128, 52.2998276, -119.9489365, 120.1486206
21: -89.1382141, 53.6313705, -88.8948593, 53.2714348, -142.4096375, 142.5262299
22: -90.9812241, 55.1813202, -90.7257538, 54.7697372, -145.7509460, 145.9070740
23: -71.4492798, 55.7965202, -71.2449799, 55.4207230, -126.8699951, 127.0414963
24: -86.1027527, 51.3888435, -85.8879166, 51.0466080, -137.1493530, 137.2767639
25: -76.2725830, 59.0684814, -76.0755005, 58.6392784, -134.9118652, 135.1439819
26: -101.2213058, 82.3688965, -100.9394226, 81.8446045, -183.0658875, 183.3082886
27: -88.7523880, 59.9402390, -88.4973450, 59.5261650, -148.2785492, 148.4375916
28: -71.7654800, 61.0620117, -71.5538788, 60.5772324, -132.3427124, 132.6158752
29: -92.3878250, 48.7685623, -92.1182480, 48.3914185, -140.7792358, 140.8868103
30: -88.5200500, 65.5275879, -88.3276825, 65.2510834, -153.7711334, 153.8552551
31: -93.3771591, 55.2662697, -93.1227875, 54.7797661, -148.1569214, 148.3890533
32: -93.2068100, 59.2374077, -92.9982605, 58.9375458, -152.1443481, 152.2356567
33: -121.7043991, 78.8361816, -121.4427338, 78.4564514, -200.1608582, 200.2789154
34: -100.7736130, 58.0767899, -100.5706177, 57.6142349, -158.3878479, 158.6473999
35: -98.9246521, 60.7466049, -98.7051697, 60.2988777, -159.2235260, 159.4517517
36: -100.4882507, 62.9793282, -100.2704086, 62.4434586, -162.9317017, 163.2497253
37: -141.7514648, 61.9997482, -141.4394531, 61.7250710, -203.4765015, 203.4391937
38: -119.7532349, 78.7866058, -119.4721909, 78.2285690, -197.9817963, 198.2587891
39: -135.0688171, 75.6079636, -134.7777252, 75.3691864, -210.4380035, 210.3856812
40: -112.0292206, 61.2892380, -111.7922745, 61.0693550, -173.0985718, 173.0815125
41: -94.2134857, 64.6405487, -93.9857788, 64.3191376, -158.5326233, 158.6263275
42: -69.2350922, 60.0032768, -69.0892410, 59.7184982, -128.9535828, 129.0925140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1866475
time: 104.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2355561, upper bound: 80.1904820
time: 134.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -106.8116760, 78.6747131, -106.4570312, 78.6513214, -185.4629974, 185.1317291
1: -59.4782028, 65.7643585, -59.2168007, 65.7478790, -125.2260818, 124.9811554
2: -53.2267151, 63.3457184, -53.0127182, 63.3239670, -116.5506821, 116.3584366
3: -58.5463524, 72.7623062, -58.3595161, 72.7323456, -131.2787018, 131.1218262
4: -66.9454651, 74.0121384, -66.6952667, 73.9902649, -140.9357300, 140.7073975
5: -60.6287231, 70.6107864, -60.4517479, 70.5782776, -131.2069855, 131.0625305
6: -95.5117035, 62.7009506, -95.4663544, 62.5693970, -158.0810852, 158.1672974
7: -70.9998932, 67.5848389, -70.7243423, 67.5647125, -138.5645905, 138.3091736
8: -77.3362427, 90.1193924, -77.0194550, 90.0834198, -167.4196625, 167.1388245
9: -65.0876465, 71.3340912, -64.8623810, 71.3087616, -136.3963928, 136.1964569
10: -95.7428665, 90.9056244, -95.4911194, 90.8607941, -186.6036224, 186.3967438
11: -91.9470673, 56.1913681, -91.8489151, 56.0806351, -148.0277100, 148.0402832
12: -91.0899429, 72.9114075, -91.0451584, 72.7718201, -163.8617554, 163.9565582
13: -97.2896500, 95.7854843, -97.0907822, 95.7257919, -193.0154419, 192.8762512
14: -142.1100159, 80.2231293, -141.8791809, 80.1889496, -222.2989655, 222.1022949
15: -75.5900726, 67.2734070, -75.4361115, 67.2369537, -142.8270264, 142.7095184
16: -96.4163208, 69.5918198, -96.1489792, 69.5586777, -165.9750061, 165.7407990
17: -139.6324158, 76.3517761, -139.3603210, 76.3056793, -215.9380951, 215.7120972
18: -91.0386124, 75.6010895, -90.9742813, 75.4132233, -166.4518433, 166.5753632
19: -72.8333435, 45.7704544, -72.7835007, 45.5916481, -118.4249878, 118.5539551
20: -67.6719055, 52.7396622, -67.6358795, 52.5736427, -120.2455368, 120.3755341
21: -89.1750793, 53.7155838, -89.1023331, 53.5498047, -142.7248840, 142.8179169
22: -91.0145721, 55.2723999, -90.9524689, 55.0708199, -146.0853882, 146.2248688
23: -71.4721069, 55.8910217, -71.4281082, 55.7363396, -127.2084351, 127.3191299
24: -86.1356201, 51.4692497, -86.0831909, 51.3150291, -137.4506531, 137.5524292
25: -76.2982483, 59.1610680, -76.2517242, 58.9452515, -135.2434692, 135.4127808
26: -101.2576981, 82.4956894, -101.1969147, 82.2707977, -183.5284729, 183.6925964
27: -88.7824631, 60.0378952, -88.7295303, 59.8546677, -148.6371307, 148.7674255
28: -71.7855988, 61.1747093, -71.7522507, 60.9517403, -132.7373352, 132.9269562
29: -92.4251709, 48.8552818, -92.3518066, 48.6816444, -141.1068115, 141.2070923
30: -88.5548477, 65.5899963, -88.4981995, 65.4671783, -154.0220337, 154.0881958
31: -93.4080429, 55.3734016, -93.3475189, 55.1344223, -148.5424652, 148.7209167
32: -93.2352448, 59.3109436, -93.1833649, 59.1797638, -152.4150085, 152.4942932
33: -121.7467270, 78.9260712, -121.6692200, 78.7738190, -200.5205383, 200.5952911
34: -100.8009338, 58.1830597, -100.7463379, 57.9741096, -158.7750397, 158.9293976
35: -98.9554825, 60.8438301, -98.8999786, 60.6300278, -159.5854950, 159.7438049
36: -100.5119171, 63.0934296, -100.4649811, 62.8276787, -163.3395996, 163.5584106
37: -141.7979126, 62.0822296, -141.7046814, 61.9948654, -203.7927856, 203.7869110
38: -119.7861023, 78.9021912, -119.7194138, 78.6274719, -198.4135742, 198.6216125
39: -135.1243744, 75.6571426, -135.0193176, 75.5396729, -210.6640472, 210.6764526
40: -112.0695877, 61.3495178, -111.9850082, 61.2681885, -173.3377533, 173.3345337
41: -94.2450943, 64.7351074, -94.1836777, 64.6326828, -158.8777618, 158.9187775
42: -69.2575836, 60.0908356, -69.2149963, 60.0041656, -129.2617493, 129.3058319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1809811, upper bound: 80.2321006
time: 145.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1809811, upper bound: 80.2355561
time: 115.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 263.50 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1627925
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1670023
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1826401
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.2355561, upper bound: 80.1863503
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1866475
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.2355561, upper bound: 80.1904820
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.1809811, upper bound: 80.2321006
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 263.50
Output dim: 1, lower bound: -80.1809811, upper bound: 80.2355561

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -105.6030655, 78.1846237, -105.6235352, 78.3397064, -183.9427795, 183.8081512
1: -58.6897316, 65.4118195, -58.6999474, 65.5332565, -124.2229691, 124.1117706
2: -52.5711670, 63.0097160, -52.5760880, 63.1264954, -115.6976547, 115.5858002
3: -57.9481812, 72.4061584, -57.9560814, 72.5073853, -130.4555664, 130.3622284
4: -66.1347046, 73.6319962, -66.1231384, 73.7565765, -139.8912811, 139.7551270
5: -60.0028419, 70.2105408, -60.0098381, 70.3187943, -130.3216400, 130.2203674
6: -95.1701355, 62.1316299, -95.2598572, 62.1713486, -157.3414764, 157.3914795
7: -70.1325684, 67.1727905, -70.1512146, 67.3178558, -137.4504242, 137.3239899
8: -76.3758011, 89.6299286, -76.4061737, 89.8095245, -166.1853333, 166.0361023
9: -64.3265381, 70.9310150, -64.3338089, 71.0612030, -135.3877411, 135.2648315
10: -94.9087601, 90.4998550, -94.9386749, 90.6035309, -185.5122986, 185.4385071
11: -91.5561981, 55.8028831, -91.5858307, 55.7878914, -147.3440857, 147.3887177
12: -90.7565536, 72.2674713, -90.8457184, 72.3539047, -163.1104584, 163.1131897
13: -96.5745010, 95.3590546, -96.5433197, 95.4041519, -191.9786530, 191.9023590
14: -141.1717224, 79.8884583, -141.1865540, 79.9691315, -221.1408539, 221.0750122
15: -75.0499268, 67.0675507, -75.0812607, 67.1119843, -142.1619110, 142.1488037
16: -95.5379486, 69.2348099, -95.5667191, 69.3190613, -164.8570099, 164.8015137
17: -138.7220459, 76.0693970, -138.7184143, 76.1082153, -214.8302307, 214.7878113
18: -90.6260300, 74.9929886, -90.7246399, 75.0251541, -165.6511688, 165.7176208
19: -72.4528122, 45.2466125, -72.5428391, 45.2221603, -117.6749573, 117.7894516
20: -67.3592758, 52.2221718, -67.4406738, 52.2222595, -119.5815277, 119.6628418
21: -88.7273407, 53.2126160, -88.8220978, 53.1972122, -141.9245453, 142.0347137
22: -90.5006104, 54.6793442, -90.6410370, 54.6807480, -145.1813507, 145.3203735
23: -71.1275940, 55.3818817, -71.1891937, 55.3448257, -126.4724197, 126.5710678
24: -85.7345963, 51.0068550, -85.8192673, 50.9798126, -136.7144012, 136.8261261
25: -75.8729858, 58.5559387, -76.0049591, 58.5522728, -134.4252625, 134.5608978
26: -100.7696457, 81.7203522, -100.8666687, 81.7213516, -182.4909668, 182.5870209
27: -88.3140488, 59.4760284, -88.4191666, 59.4444771, -147.7585144, 147.8952026
28: -71.3872528, 60.5064697, -71.4886551, 60.4795837, -131.8668213, 131.9951172
29: -91.9191284, 48.3395271, -92.0344696, 48.3160744, -140.2351837, 140.3739929
30: -88.2084808, 65.1915131, -88.2669220, 65.1929932, -153.4014740, 153.4584351
31: -92.8909149, 54.6797905, -93.0380707, 54.6768112, -147.5677185, 147.7178650
32: -92.8660736, 58.8060837, -92.9427338, 58.8506165, -151.7166901, 151.7488098
33: -121.2911530, 78.2901001, -121.3770599, 78.3452759, -199.6364136, 199.6671448
34: -100.4164124, 57.4728851, -100.5131226, 57.4999542, -157.9163666, 157.9860077
35: -98.5215683, 60.1405678, -98.6404114, 60.1847229, -158.7062683, 158.7809753
36: -100.0578156, 62.2684784, -100.2026520, 62.3125877, -162.3703613, 162.4711304
37: -141.2674866, 61.5842361, -141.3648224, 61.6286545, -202.8961487, 202.9490662
38: -119.2176285, 78.0044098, -119.3876801, 78.0823822, -197.3000183, 197.3920898
39: -134.6271667, 75.2134323, -134.7064362, 75.2888794, -209.9160461, 209.9198608
40: -111.6671829, 60.9212265, -111.7370834, 60.9870186, -172.6541901, 172.6583099
41: -93.8853073, 64.2378387, -93.9350586, 64.2318954, -158.1172028, 158.1728821
42: -69.0314178, 59.6279640, -69.0583496, 59.6362953, -128.6677094, 128.6863098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 733

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1627925
time: 116.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1627925
time: 121.42 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -106.1799850, 78.3716507, -105.7934265, 78.3530655, -184.5330505, 184.1650696
1: -59.0589218, 65.5472870, -58.8102188, 65.5412598, -124.6001816, 124.3575058
2: -52.8819962, 63.1361351, -52.6694832, 63.1360207, -116.0180054, 115.8056183
3: -58.2402153, 72.5454788, -58.0439339, 72.5221786, -130.7623901, 130.5894165
4: -66.5225449, 73.7728195, -66.2387695, 73.7677460, -140.2902832, 140.0115967
5: -60.3202324, 70.3639526, -60.1052170, 70.3330917, -130.6533203, 130.4691620
6: -95.3067932, 62.4338760, -95.2781448, 62.2604408, -157.5672302, 157.7120056
7: -70.5535812, 67.3348083, -70.2785797, 67.3294525, -137.8830261, 137.6133881
8: -76.8338165, 89.8164062, -76.5434952, 89.8259735, -166.6597900, 166.3598938
9: -64.6919556, 71.0889435, -64.4420929, 71.0749207, -135.7668762, 135.5310211
10: -95.3069534, 90.6645355, -95.0556335, 90.6266861, -185.9336395, 185.7201538
11: -91.6875000, 55.9764786, -91.6198120, 55.8380470, -147.5255432, 147.5962830
12: -90.9035110, 72.6218872, -90.8656921, 72.4562683, -163.3597565, 163.4875793
13: -96.9085541, 95.5056534, -96.6407471, 95.4321747, -192.3407288, 192.1463928
14: -141.6310120, 80.0192413, -141.3197632, 79.9844818, -221.6154938, 221.3389893
15: -75.3042984, 67.1247101, -75.1530457, 67.1246185, -142.4289246, 142.2777557
16: -95.9345169, 69.3844452, -95.6837311, 69.3370972, -165.2716064, 165.0681763
17: -139.1365967, 76.1754074, -138.8382263, 76.1306915, -215.2672882, 215.0136414
18: -90.7905731, 75.2868118, -90.7517014, 75.1123581, -165.9029236, 166.0385132
19: -72.5826874, 45.4859161, -72.5632477, 45.2950287, -117.8777084, 118.0491638
20: -67.4714508, 52.4592018, -67.4585419, 52.2928886, -119.7643356, 119.9177399
21: -88.8855591, 53.4413528, -88.8525848, 53.2653923, -142.1509552, 142.2939453
22: -90.6809921, 54.9500313, -90.6699905, 54.7624702, -145.4434662, 145.6200256
23: -71.2462769, 55.6163826, -71.2074966, 55.4143906, -126.6606598, 126.8238831
24: -85.8662109, 51.2142181, -85.8441772, 51.0415497, -136.9077454, 137.0583954
25: -76.0206604, 58.8232994, -76.0268860, 58.6320877, -134.6527405, 134.8501892
26: -100.9725876, 82.1029358, -100.8946609, 81.8364944, -182.8090820, 182.9975891
27: -88.4686584, 59.7313423, -88.4428635, 59.5196571, -147.9883118, 148.1742096
28: -71.5231781, 60.8124657, -71.5044098, 60.5707169, -132.0939026, 132.3168640
29: -92.0919876, 48.5709953, -92.0665207, 48.3853607, -140.4773560, 140.6375122
30: -88.3185654, 65.3601685, -88.2937317, 65.2407913, -153.5593414, 153.6539001
31: -93.0697632, 54.9997253, -93.0644989, 54.7730484, -147.8428040, 148.0642090
32: -93.0078735, 59.0704727, -92.9632568, 58.9284630, -151.9363403, 152.0337219
33: -121.4872131, 78.6297607, -121.4101410, 78.4453201, -199.9325256, 200.0398865
34: -100.5744019, 57.8269882, -100.5352554, 57.6064529, -158.1808319, 158.3622437
35: -98.7002335, 60.4953766, -98.6652069, 60.2913742, -158.9915924, 159.1605835
36: -100.2411118, 62.6804886, -100.2221375, 62.4369507, -162.6780396, 162.9026184
37: -141.4985657, 61.8841934, -141.4019318, 61.7186584, -203.2172241, 203.2861176
38: -119.4582138, 78.4537201, -119.4160614, 78.2167511, -197.6749268, 197.8697815
39: -134.8282471, 75.4559479, -134.7472839, 75.3614807, -210.1897125, 210.2032166
40: -111.8350296, 61.1781006, -111.7679672, 61.0626564, -172.8976898, 172.9460602
41: -94.0348053, 64.5101776, -93.9579697, 64.3126678, -158.3474731, 158.4681396
42: -69.1270294, 59.8781319, -69.0747375, 59.7101669, -128.8371887, 128.9528656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2321007, upper bound: 80.1112617
time: 102.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2321007, upper bound: 80.1112617
time: 102.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -105.7945023, 78.2027206, -106.2693176, 78.5763779, -184.3708801, 184.4720306
1: -58.8078880, 65.4225464, -59.0977936, 65.6951904, -124.5030823, 124.5203400
2: -52.6723938, 63.0215340, -52.9130898, 63.2729492, -115.9453354, 115.9346237
3: -58.0406075, 72.4237366, -58.2660103, 72.6776123, -130.7182159, 130.6897278
4: -66.2675095, 73.6457367, -66.5712204, 73.9311981, -140.1987000, 140.2169495
5: -60.1040688, 70.2289886, -60.3510475, 70.5178833, -130.6219482, 130.5800323
6: -95.1953583, 62.2213860, -95.4163589, 62.4706345, -157.6659851, 157.6377258
7: -70.2637100, 67.1857376, -70.5893021, 67.5022583, -137.7659607, 137.7750397
8: -76.5130463, 89.6487350, -76.8712769, 90.0085678, -166.5216064, 166.5200043
9: -64.4463577, 70.9468231, -64.7432480, 71.2468872, -135.6932373, 135.6900635
10: -95.0307007, 90.5271454, -95.3609924, 90.7981644, -185.8288574, 185.8881378
11: -91.6012115, 55.8727188, -91.7883606, 56.0233536, -147.6245728, 147.6610718
12: -90.7802582, 72.3544464, -90.9978027, 72.6549454, -163.4352112, 163.3522339
13: -96.7039032, 95.3939133, -96.9831390, 95.6540451, -192.3579407, 192.3770447
14: -141.3299408, 79.9072495, -141.7287598, 80.1379242, -221.4678192, 221.6360168
15: -75.1265564, 67.0876160, -75.3518677, 67.2027588, -142.3293152, 142.4394684
16: -95.6687012, 69.2550354, -96.0139771, 69.5045242, -165.1732178, 165.2690125
17: -138.8705444, 76.0959167, -139.2264709, 76.2606125, -215.1311646, 215.3223877
18: -90.6579056, 75.0766907, -90.9116592, 75.3167191, -165.9746246, 165.9883423
19: -72.4779358, 45.3356667, -72.7220306, 45.5151596, -117.9930954, 118.0576935
20: -67.3822937, 52.3048401, -67.5851364, 52.4965744, -119.8788605, 119.8899689
21: -88.7645340, 53.2961159, -89.0299759, 53.4761162, -142.2406464, 142.3260803
22: -90.5343781, 54.7695274, -90.8684082, 54.9817619, -145.5161438, 145.6379395
23: -71.1507797, 55.4758759, -71.3724365, 55.6609879, -126.8117676, 126.8483124
24: -85.7676926, 51.0866089, -86.0145035, 51.2485504, -137.0162354, 137.1011047
25: -75.8988800, 58.6475105, -76.1813660, 58.8582954, -134.7571716, 134.8288727
26: -100.8067703, 81.8473969, -101.1251755, 82.1481171, -182.9548950, 182.9725647
27: -88.3443756, 59.5734329, -88.6514282, 59.7737961, -148.1181641, 148.2248535
28: -71.4077377, 60.6188507, -71.6873932, 60.8547211, -132.2624512, 132.3062286
29: -91.9565506, 48.4257393, -92.2683334, 48.6064987, -140.5630493, 140.6940765
30: -88.2434769, 65.2532196, -88.4375534, 65.4095612, -153.6530457, 153.6907654
31: -92.9221649, 54.7859726, -93.2627258, 55.0317993, -147.9539642, 148.0487061
32: -92.8945389, 58.8794632, -93.1282425, 59.0933952, -151.9879303, 152.0077057
33: -121.3342590, 78.3805466, -121.6038971, 78.6626282, -199.9968872, 199.9844360
34: -100.4440842, 57.5809059, -100.6911926, 57.8601303, -158.3042145, 158.2720947
35: -98.5529633, 60.2381210, -98.8358154, 60.5160942, -159.0690613, 159.0739288
36: -100.0820084, 62.3828850, -100.3980179, 62.6970482, -162.7790527, 162.7808990
37: -141.3143921, 61.6641159, -141.6316071, 61.8982925, -203.2126770, 203.2957153
38: -119.2509689, 78.1216812, -119.6382065, 78.4815826, -197.7325439, 197.7598724
39: -134.6842499, 75.2616730, -134.9483948, 75.4597244, -210.1439819, 210.2100677
40: -111.7075653, 60.9820175, -111.9295883, 61.1875267, -172.8950958, 172.9116058
41: -93.9170532, 64.3332977, -94.1341095, 64.5463867, -158.4634399, 158.4674072
42: -69.0537567, 59.7138519, -69.1841736, 59.9231453, -128.9768829, 128.8980255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 733

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1826401
time: 1680.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1826401
time: 110.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 1793.47 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1793.47
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1627925
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1793.47
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1627925
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1793.47
Output dim: 1, lower bound: -80.2321007, upper bound: 80.1112617
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1793.47
Output dim: 1, lower bound: -80.2321007, upper bound: 80.1112617
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1793.47
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1826401
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1793.47
Output dim: 1, lower bound: -80.1569851, upper bound: 80.1826401
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1793.47
Output dim: 1, lower bound: -80.2355561, upper bound: 80.1863503
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1793.47
Output dim: 1, lower bound: -80.1809811, upper bound: 80.1866475
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1793.47
Output dim: 1, lower bound: -80.2355561, upper bound: 80.1904820
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1793.47
Output dim: 1, lower bound: -80.1809811, upper bound: 80.2321006
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1793.47
Output dim: 1, lower bound: -80.1809811, upper bound: 80.2355561
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=124.9945068359375
rel_dist={1: [-80.2411023357962, 80.2411023398264]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4764032, upper bound: 79.4334929
time: 96.02 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4764032, upper bound: 79.4764031
time: 261.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 357.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 357.47
Output dim: 1, lower bound: -79.4764032, upper bound: 79.4334929
IS_A2, status: Status.UNKNOWN, split count: 1, time: 357.47
Output dim: 1, lower bound: -79.4764032, upper bound: 79.4764031

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -106.3987808, 78.3931427, -106.4618759, 78.5752563, -184.9740295, 184.8550110
1: -59.1949806, 65.5602417, -59.2249794, 65.6923141, -124.8872986, 124.7852173
2: -52.9987984, 63.1506500, -53.0229225, 63.2731552, -116.2719574, 116.1735687
3: -58.3475113, 72.5665359, -58.3688202, 72.6847076, -131.0322266, 130.9353638
4: -66.6753998, 73.7898865, -66.7070541, 73.9319305, -140.6073303, 140.4969330
5: -60.4376793, 70.3861237, -60.4606857, 70.5226288, -130.9602966, 130.8468018
6: -95.3362808, 62.5405197, -95.4315262, 62.5794296, -157.9157104, 157.9720459
7: -70.7053680, 67.3509064, -70.7343597, 67.5012894, -138.2066650, 138.0852661
8: -76.9942551, 89.8392029, -77.0299377, 90.0128403, -167.0070953, 166.8691406
9: -64.8290787, 71.1079102, -64.8642807, 71.2508087, -136.0798798, 135.9721985
10: -95.4483795, 90.6976547, -95.4921417, 90.8171158, -186.2654724, 186.1897736
11: -91.7407379, 56.0580711, -91.8259964, 56.0861702, -147.8269043, 147.8840637
12: -90.9319611, 72.7274628, -91.0157623, 72.7734146, -163.7053833, 163.7432251
13: -97.0572739, 95.5465546, -97.1007690, 95.6774597, -192.7347412, 192.6473236
14: -141.8136902, 80.0412674, -141.8813171, 80.1469116, -221.9606018, 221.9225769
15: -75.3955841, 67.1489944, -75.4386215, 67.2154236, -142.6110077, 142.5876160
16: -96.0858536, 69.4087601, -96.1461334, 69.5171967, -165.6030273, 165.5549011
17: -139.3055725, 76.2069168, -139.3646851, 76.2812424, -215.5867920, 215.5715942
18: -90.8292084, 75.3867340, -90.9372559, 75.4173965, -166.2466125, 166.3239746
19: -72.6127319, 45.5880623, -72.7357864, 45.6030083, -118.2157135, 118.3238525
20: -67.4992828, 52.5560455, -67.5989227, 52.5823708, -120.0816498, 120.1549606
21: -88.9300766, 53.5384483, -89.0569382, 53.5601807, -142.4902496, 142.5953827
22: -90.7209625, 55.0547943, -90.8869476, 55.0791626, -145.8001251, 145.9417419
23: -71.2746277, 55.7239151, -71.3871765, 55.7460747, -127.0206985, 127.1110840
24: -85.9065552, 51.3063354, -86.0375061, 51.3245201, -137.2310791, 137.3438416
25: -76.0526505, 58.9300232, -76.1964722, 58.9550323, -135.0076904, 135.1264954
26: -101.0170441, 82.2497406, -101.1492233, 82.2803879, -183.2974243, 183.3989563
27: -88.5054169, 59.8431320, -88.6669998, 59.8661003, -148.3715057, 148.5101318
28: -71.5487061, 60.9409828, -71.6946182, 60.9647980, -132.5135040, 132.6355896
29: -92.1360092, 48.6692810, -92.2915649, 48.6896095, -140.8256226, 140.9608459
30: -88.3624573, 65.4331512, -88.4668655, 65.4694061, -153.8318634, 153.9000092
31: -93.1065674, 55.1229706, -93.2786560, 55.1453133, -148.2518768, 148.4016113
32: -93.0414886, 59.1597519, -93.1459427, 59.1899452, -152.2314301, 152.3056946
33: -121.5373611, 78.7372742, -121.6382675, 78.7749786, -200.3123474, 200.3755188
34: -100.6080170, 57.9516373, -100.7121048, 57.9786148, -158.5866089, 158.6637421
35: -98.7382889, 60.6092262, -98.8589020, 60.6348076, -159.3730927, 159.4681244
36: -100.2701111, 62.8136902, -100.4104233, 62.8344765, -163.1045837, 163.2241211
37: -141.5543823, 61.9797630, -141.6702271, 62.0021095, -203.5564880, 203.6499939
38: -119.4981003, 78.5923767, -119.6606216, 78.6306000, -198.1286926, 198.2529907
39: -134.8947449, 75.5172653, -134.9955139, 75.5458527, -210.4405975, 210.5127869
40: -111.8822021, 61.2507858, -111.9616623, 61.2836456, -173.1658325, 173.2124481
41: -94.0716019, 64.6205139, -94.1566086, 64.6456757, -158.7172546, 158.7771301
42: -69.1534271, 59.9795609, -69.2014618, 60.0134048, -129.1668396, 129.1810303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4133010
time: 110.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4301803
time: 123.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -106.8154907, 78.6753693, -106.4816895, 78.6549225, -185.4703827, 185.1570587
1: -59.4805145, 65.7647858, -59.2343750, 65.7501984, -125.2307129, 124.9991531
2: -53.2288322, 63.3461914, -53.0293312, 63.3268051, -116.5556335, 116.3755112
3: -58.5482407, 72.7629242, -58.3744392, 72.7361755, -131.2844086, 131.1373596
4: -66.9480667, 74.0127106, -66.7155533, 73.9937744, -140.9418335, 140.7282715
5: -60.6308441, 70.6114120, -60.4658508, 70.5821228, -131.2129669, 131.0772552
6: -95.5125351, 62.7040367, -95.4721680, 62.5903587, -158.1028900, 158.1762085
7: -71.0024414, 67.5852814, -70.7421036, 67.5670776, -138.5695190, 138.3273926
8: -77.3389282, 90.1202164, -77.0409241, 90.0884552, -167.4273834, 167.1611328
9: -65.0898590, 71.3347473, -64.8762894, 71.3130798, -136.4029236, 136.2110291
10: -95.7453384, 90.9067078, -95.5068970, 90.8679657, -186.6133118, 186.4136047
11: -91.9487000, 56.1931839, -91.8602829, 56.0938492, -148.0425415, 148.0534668
12: -91.0908203, 72.9143219, -91.0511932, 72.7895355, -163.8803406, 163.9655151
13: -97.2929535, 95.7867126, -97.1120300, 95.7336121, -193.0265656, 192.8987122
14: -142.1135254, 80.2237549, -141.9011078, 80.1929169, -222.3064423, 222.1248627
15: -75.5925140, 67.2743225, -75.4531326, 67.2430267, -142.8355408, 142.7274475
16: -96.4193039, 69.5926208, -96.1674118, 69.5638657, -165.9831543, 165.7600403
17: -139.6358032, 76.3525314, -139.3815002, 76.3103027, -215.9460907, 215.7340393
18: -91.0398788, 75.6030121, -90.9829788, 75.4272079, -166.4670715, 166.5859985
19: -72.8342285, 45.7722473, -72.7887573, 45.6060562, -118.4402771, 118.5610046
20: -67.6727600, 52.7418060, -67.6412201, 52.5890617, -120.2618256, 120.3830261
21: -89.1764374, 53.7175903, -89.1108170, 53.5655632, -142.7419891, 142.8283997
22: -91.0156021, 55.2743340, -90.9583359, 55.0866776, -146.1022797, 146.2326660
23: -71.4732056, 55.8929939, -71.4352722, 55.7518158, -127.2250214, 127.3282547
24: -86.1371002, 51.4709854, -86.0933990, 51.3288994, -137.4659729, 137.5643921
25: -76.2993317, 59.1632042, -76.2587738, 58.9621658, -135.2614899, 135.4219818
26: -101.2590714, 82.4985733, -101.2055130, 82.2879028, -183.5469666, 183.7040863
27: -88.7836151, 60.0400696, -88.7368317, 59.8715858, -148.6551971, 148.7768860
28: -71.7865219, 61.1770782, -71.7579346, 60.9704094, -132.7569275, 132.9350128
29: -92.4262161, 48.8569946, -92.3580093, 48.6955643, -141.1217651, 141.2149963
30: -88.5565414, 65.5917969, -88.5100937, 65.4805145, -154.0370483, 154.1018982
31: -93.4090576, 55.3755188, -93.3536835, 55.1512871, -148.5603485, 148.7292023
32: -93.2362366, 59.3136864, -93.1905060, 59.1992226, -152.4354553, 152.5041809
33: -121.7482834, 78.9277878, -121.6797104, 78.7872009, -200.5354614, 200.6074982
34: -100.8021698, 58.1851883, -100.7545929, 57.9865723, -158.7887421, 158.9397888
35: -98.9569168, 60.8457680, -98.9095535, 60.6423798, -159.5992737, 159.7553253
36: -100.5130081, 63.0955429, -100.4719315, 62.8404045, -163.3533936, 163.5674744
37: -141.7996826, 62.0837936, -141.7162476, 62.0085335, -203.8081970, 203.8000488
38: -119.7875290, 78.9047165, -119.7292099, 78.6430664, -198.4305878, 198.6339264
39: -135.1265564, 75.6590652, -135.0340271, 75.5543137, -210.6808777, 210.6930847
40: -112.0707779, 61.3525848, -111.9932098, 61.2901001, -173.3608704, 173.3457947
41: -94.2461090, 64.7375641, -94.1908340, 64.6511688, -158.8972778, 158.9284058
42: -69.2583313, 60.0934944, -69.2200165, 60.0210075, -129.2793274, 129.3135071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4351627
time: 107.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4751897
time: 104.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 214.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 214.87
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4133010
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 214.87
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4301803
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 214.87
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4351627
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 214.87
Output dim: 1, lower bound: -79.4751898, upper bound: 79.4751897

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -106.1663818, 78.3708649, -105.7888641, 78.3336487, -184.5000305, 184.1596985
1: -59.0509262, 65.5469208, -58.8084755, 65.5271072, -124.5780334, 124.3553925
2: -52.8751526, 63.1359596, -52.6681442, 63.1229591, -115.9981079, 115.8041077
3: -58.2344589, 72.5448303, -58.0428238, 72.5096664, -130.7440948, 130.5876465
4: -66.5131683, 73.7727356, -66.2369232, 73.7526398, -140.2658081, 140.0096436
5: -60.3139267, 70.3634491, -60.1040077, 70.3186340, -130.6325684, 130.4674530
6: -95.3051147, 62.4298286, -95.2680130, 62.2574615, -157.5625610, 157.6978455
7: -70.5454330, 67.3349304, -70.2771454, 67.3134918, -137.8589172, 137.6120758
8: -76.8262787, 89.8157806, -76.5414963, 89.8075256, -166.6338043, 166.3572693
9: -64.6837463, 71.0882874, -64.4396896, 71.0596771, -135.7434235, 135.5279846
10: -95.2995758, 90.6638184, -95.0523758, 90.6141968, -185.9137573, 185.7161865
11: -91.6853485, 55.9722061, -91.6113358, 55.8357964, -147.5211487, 147.5835419
12: -90.9026260, 72.6203461, -90.8568954, 72.4525909, -163.3552246, 163.4772339
13: -96.8983307, 95.5037384, -96.6373901, 95.4184265, -192.3167419, 192.1411133
14: -141.6209106, 80.0182800, -141.3143616, 79.9734573, -221.5943604, 221.3326416
15: -75.3007965, 67.1238251, -75.1494293, 67.1177673, -142.4185638, 142.2732391
16: -95.9267731, 69.3837280, -95.6789322, 69.3256073, -165.2523804, 165.0626526
17: -139.1245728, 76.1741791, -138.8334656, 76.1231995, -215.2477722, 215.0076141
18: -90.7897110, 75.2835464, -90.7404709, 75.1102905, -165.8999939, 166.0240173
19: -72.5817871, 45.4789009, -72.5502472, 45.2943802, -117.8761597, 118.0291443
20: -67.4707794, 52.4543953, -67.4480591, 52.2910919, -119.7618713, 119.9024506
21: -88.8840714, 53.4355850, -88.8393555, 53.2639999, -142.1480713, 142.2749329
22: -90.6796188, 54.9440346, -90.6525497, 54.7609253, -145.4405365, 145.5965881
23: -71.2457581, 55.6085472, -71.1956482, 55.4130020, -126.6587296, 126.8041992
24: -85.8652802, 51.2080574, -85.8305206, 51.0404549, -136.9057312, 137.0385590
25: -76.0204086, 58.8173294, -76.0118103, 58.6305275, -134.6509399, 134.8291321
26: -100.9712296, 82.0946655, -100.8807297, 81.8347626, -182.8059998, 182.9754028
27: -88.4676361, 59.7232170, -88.4257965, 59.5181961, -147.9858398, 148.1490173
28: -71.5231705, 60.8032494, -71.4889526, 60.5693474, -132.0925140, 132.2921753
29: -92.0902328, 48.5636597, -92.0503387, 48.3840675, -140.4743042, 140.6139832
30: -88.3185959, 65.3563004, -88.2830887, 65.2376175, -153.5562134, 153.6393890
31: -93.0681000, 54.9928093, -93.0462189, 54.7719421, -147.8400421, 148.0390320
32: -93.0063095, 59.0687065, -92.9522095, 58.9263229, -151.9326324, 152.0209045
33: -121.4844131, 78.6266327, -121.3996124, 78.4425964, -199.9270020, 200.0262451
34: -100.5735397, 57.8205070, -100.5240936, 57.6049423, -158.1784668, 158.3446045
35: -98.6992722, 60.4905777, -98.6525650, 60.2899551, -158.9892273, 159.1431427
36: -100.2402115, 62.6748505, -100.2070236, 62.4362793, -162.6764679, 162.8818512
37: -141.4966431, 61.8816147, -141.3898926, 61.7174988, -203.2140808, 203.2715149
38: -119.4567490, 78.4495773, -119.3985825, 78.2144318, -197.6711731, 197.8481445
39: -134.8246918, 75.4571838, -134.7370911, 75.3594437, -210.1841125, 210.1942749
40: -111.8325958, 61.1746254, -111.7599564, 61.0597725, -172.8923645, 172.9345856
41: -94.0325546, 64.5034027, -93.9490051, 64.3110504, -158.3435974, 158.4524078
42: -69.1257019, 59.8734016, -69.0698624, 59.7076263, -128.8333282, 128.9432526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4074596
time: 327.91 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4121865
time: 127.64 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -106.3929520, 78.3920822, -106.4354401, 78.5703888, -184.9633484, 184.8275146
1: -59.1907997, 65.5595169, -59.2065201, 65.6891022, -124.8798981, 124.7660370
2: -52.9949913, 63.1498489, -53.0054588, 63.2694855, -116.2644730, 116.1553040
3: -58.3440552, 72.5654144, -58.3531532, 72.6799622, -131.0240173, 130.9185638
4: -66.6707611, 73.7888870, -66.6856766, 73.9273911, -140.5981445, 140.4745636
5: -60.4340248, 70.3850403, -60.4456139, 70.5177307, -130.9517517, 130.8306427
6: -95.3347473, 62.5356750, -95.4246979, 62.5571747, -157.8919220, 157.9603729
7: -70.7007523, 67.3500977, -70.7155151, 67.4978638, -138.1986084, 138.0656128
8: -76.9893494, 89.8377991, -77.0073547, 90.0066452, -166.9959717, 166.8451538
9: -64.8258362, 71.1067200, -64.8495865, 71.2454376, -136.0712738, 135.9562988
10: -95.4446793, 90.6958084, -95.4753494, 90.8088837, -186.2535706, 186.1711578
11: -91.7379456, 56.0549316, -91.8133392, 56.0720596, -147.8099976, 147.8682556
12: -90.9303665, 72.7233810, -91.0087891, 72.7547302, -163.6850891, 163.7321625
13: -97.0515976, 95.5445404, -97.0780792, 95.6683960, -192.7199707, 192.6226196
14: -141.8083649, 80.0401764, -141.8573303, 80.1419830, -221.9503479, 221.8975067
15: -75.3915100, 67.1474304, -75.4206085, 67.2083588, -142.5998688, 142.5680237
16: -96.0814896, 69.4073334, -96.1263504, 69.5110779, -165.5925598, 165.5336914
17: -139.3003998, 76.2056274, -139.3415527, 76.2753525, -215.5757446, 215.5471802
18: -90.8270416, 75.3831482, -90.9274750, 75.4026184, -166.2296600, 166.3106232
19: -72.6113205, 45.5847855, -72.7294235, 45.5879974, -118.1993179, 118.3141937
20: -67.4978943, 52.5521927, -67.5926666, 52.5659981, -120.0638885, 120.1448593
21: -88.9279404, 53.5348206, -89.0472336, 53.5435715, -142.4714966, 142.5820465
22: -90.7193069, 55.0512085, -90.8796844, 55.0625687, -145.7818756, 145.9308777
23: -71.2728271, 55.7203941, -71.3789368, 55.7298241, -127.0026550, 127.0993347
24: -85.9040375, 51.3031845, -86.0260391, 51.3100052, -137.2140503, 137.3292236
25: -76.0508270, 58.9261475, -76.1882935, 58.9372787, -134.9880981, 135.1144409
26: -101.0148163, 82.2455826, -101.1393356, 82.2621078, -183.2769165, 183.3849182
27: -88.5035248, 59.8392830, -88.6583557, 59.8483505, -148.3518677, 148.4976196
28: -71.5472412, 60.9367447, -71.6878204, 60.9452133, -132.4924622, 132.6245422
29: -92.1342468, 48.6661568, -92.2839050, 48.6750336, -140.8092804, 140.9500580
30: -88.3596268, 65.4300079, -88.4537964, 65.4550858, -153.8147125, 153.8838043
31: -93.1048660, 55.1190567, -93.2711334, 55.1276588, -148.2325134, 148.3901978
32: -93.0396805, 59.1553154, -93.1377563, 59.1696281, -152.2093048, 152.2930756
33: -121.5348434, 78.7340393, -121.6265945, 78.7605438, -200.2953796, 200.3606262
34: -100.6059113, 57.9485893, -100.7025070, 57.9652557, -158.5711670, 158.6510925
35: -98.7359619, 60.6062698, -98.8481598, 60.6216736, -159.3576050, 159.4544220
36: -100.2683334, 62.8106689, -100.4023819, 62.8211098, -163.0894470, 163.2130432
37: -141.5514526, 61.9758949, -141.6569214, 61.9877930, -203.5392456, 203.6328125
38: -119.4956589, 78.5887299, -119.6495285, 78.6140518, -198.1097107, 198.2382507
39: -134.8911896, 75.5138397, -134.9791107, 75.5303040, -210.4214935, 210.4929504
40: -111.8801346, 61.2457161, -111.9521942, 61.2600403, -173.1401520, 173.1979065
41: -94.0697784, 64.6162872, -94.1482620, 64.6261597, -158.6959381, 158.7645416
42: -69.1521454, 59.9754028, -69.1955566, 59.9954872, -129.1476135, 129.1709595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4247705
time: 130.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4290534
time: 118.66 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -106.5849762, 78.6531830, -105.8100052, 78.4136658, -184.9986267, 184.4631958
1: -59.3357849, 65.7516708, -58.8182182, 65.5855026, -124.9212646, 124.5698776
2: -53.1048546, 63.3316193, -52.6751366, 63.1770325, -116.2818909, 116.0067596
3: -58.4349098, 72.7413330, -58.0491753, 72.5612259, -130.9961395, 130.7904968
4: -66.7858353, 73.9956360, -66.2467194, 73.8147354, -140.6005554, 140.2423401
5: -60.5067711, 70.5889664, -60.1101112, 70.3784637, -130.8852386, 130.6990814
6: -95.4811630, 62.5955582, -95.3087616, 62.2697334, -157.7508850, 157.9043274
7: -70.8415604, 67.5693359, -70.2853546, 67.3795624, -138.2211304, 137.8546906
8: -77.1704559, 90.0969086, -76.5533142, 89.8835297, -167.0539551, 166.6502228
9: -64.9454880, 71.3150558, -64.4526367, 71.1219788, -136.0674744, 135.7677002
10: -95.5960159, 90.8731308, -95.0683670, 90.6655731, -186.2615967, 185.9414978
11: -91.8931732, 56.1071129, -91.6444321, 55.8450394, -147.7382202, 147.7515411
12: -91.0615463, 72.8098068, -90.8925705, 72.4697571, -163.5313110, 163.7023621
13: -97.1346970, 95.7442780, -96.6508789, 95.4743500, -192.6090393, 192.3951569
14: -141.9216614, 80.2009125, -141.3360748, 80.0173645, -221.9390106, 221.5369873
15: -75.4979248, 67.2492523, -75.1650848, 67.1441040, -142.6420288, 142.4143372
16: -96.2606201, 69.5675964, -95.7005310, 69.3726044, -165.6332245, 165.2681122
17: -139.4563599, 76.3202362, -138.8518219, 76.1506119, -215.6069641, 215.1720581
18: -91.0007782, 75.4998474, -90.7867203, 75.1210709, -166.1218567, 166.2865601
19: -72.8035660, 45.6630287, -72.6034012, 45.2984390, -118.1020050, 118.2664337
20: -67.6445389, 52.6402473, -67.4909439, 52.2988052, -119.9433441, 120.1311951
21: -89.1308441, 53.6146774, -88.8935623, 53.2705994, -142.4014435, 142.5082397
22: -90.9745560, 55.1632767, -90.7243423, 54.7689209, -145.7434692, 145.8876190
23: -71.4446182, 55.7778664, -71.2438583, 55.4199371, -126.8645554, 127.0217285
24: -86.0961151, 51.3729286, -85.8866272, 51.0458908, -137.1419983, 137.2595520
25: -76.2674332, 59.0501099, -76.0743332, 58.6384125, -134.9058380, 135.1244507
26: -101.2139664, 82.3437195, -100.9381409, 81.8433533, -183.0573120, 183.2818604
27: -88.7463303, 59.9209251, -88.4959259, 59.5252914, -148.2716217, 148.4168549
28: -71.7613907, 61.0396919, -71.5527039, 60.5762444, -132.3376160, 132.5923920
29: -92.3803864, 48.7514572, -92.1168365, 48.3907547, -140.7711487, 140.8682709
30: -88.5129776, 65.5152283, -88.3263550, 65.2500076, -153.7629852, 153.8415680
31: -93.3709869, 55.2451096, -93.1213760, 54.7789612, -148.1499329, 148.3664703
32: -93.2011261, 59.2226257, -92.9971924, 58.9366341, -152.1377563, 152.2198181
33: -121.6959000, 78.8185577, -121.4414749, 78.4553833, -200.1512756, 200.2600403
34: -100.7681122, 58.0557327, -100.5694504, 57.6133194, -158.3814240, 158.6251831
35: -98.9184189, 60.7273788, -98.7039185, 60.2980194, -159.2164307, 159.4313049
36: -100.4834671, 62.9567566, -100.2692871, 62.4427338, -162.9261780, 163.2260437
37: -141.7421265, 61.9836311, -141.4377441, 61.7242699, -203.4664001, 203.4213715
38: -119.7466049, 78.7637558, -119.4708557, 78.2275391, -197.9741364, 198.2345886
39: -135.0576630, 75.5981064, -134.7759399, 75.3682175, -210.4258423, 210.3740234
40: -112.0212097, 61.2770042, -111.7909470, 61.0677490, -173.0889587, 173.0679474
41: -94.2071686, 64.6219635, -93.9846954, 64.3180847, -158.5252380, 158.6066589
42: -69.2306061, 59.9858589, -69.0883636, 59.7170753, -128.9476624, 129.0742188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4286816
time: 140.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4333430
time: 95.97 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -106.8089066, 78.6742401, -106.4548950, 78.6499786, -185.4588928, 185.1291351
1: -59.4765091, 65.7640381, -59.2158127, 65.7469177, -125.2234192, 124.9798431
2: -53.2251625, 63.3453712, -53.0116768, 63.3230782, -116.5482407, 116.3570480
3: -58.5449638, 72.7618866, -58.3585892, 72.7313614, -131.2763214, 131.1204834
4: -66.9435120, 74.0117188, -66.6939392, 73.9891205, -140.9326324, 140.7056580
5: -60.6271591, 70.6103058, -60.4505539, 70.5771866, -131.2043457, 131.0608521
6: -95.5111237, 62.6986504, -95.4652405, 62.5678253, -158.0789337, 158.1638794
7: -70.9980469, 67.5844955, -70.7230988, 67.5636215, -138.5616760, 138.3075867
8: -77.3342896, 90.1187897, -77.0182190, 90.0821075, -167.4163971, 167.1369934
9: -65.0860443, 71.3335800, -64.8614120, 71.3076019, -136.3936310, 136.1949768
10: -95.7409821, 90.9048538, -95.4898834, 90.8595352, -186.6005249, 186.3947296
11: -91.9458542, 56.1900253, -91.8474808, 56.0794792, -148.0253296, 148.0375061
12: -91.0893097, 72.9091949, -91.0440674, 72.7705078, -163.8598175, 163.9532623
13: -97.2872162, 95.7846222, -97.0888672, 95.7244110, -193.0116272, 192.8734741
14: -142.1074524, 80.2226562, -141.8766022, 80.1879578, -222.2954102, 222.0992584
15: -75.5882111, 67.2727737, -75.4348297, 67.2358398, -142.8240356, 142.7076111
16: -96.4141464, 69.5912094, -96.1474380, 69.5576630, -165.9718018, 165.7386475
17: -139.6298828, 76.3511734, -139.3580780, 76.3043213, -215.9342041, 215.7092285
18: -91.0376740, 75.5996323, -90.9730377, 75.4122467, -166.4499207, 166.5726624
19: -72.8326721, 45.7690887, -72.7822800, 45.5908165, -118.4234695, 118.5513611
20: -67.6712646, 52.7380753, -67.6348114, 52.5724182, -120.2436829, 120.3728790
21: -89.1740875, 53.7140617, -89.1008987, 53.5487289, -142.7228088, 142.8149567
22: -91.0138016, 55.2709732, -90.9509125, 55.0699348, -146.0837402, 146.2218933
23: -71.4713287, 55.8895149, -71.4268951, 55.7353363, -127.2066498, 127.3164062
24: -86.1345673, 51.4679489, -86.0817566, 51.3141365, -137.4487000, 137.5497131
25: -76.2974243, 59.1595306, -76.2504272, 58.9442520, -135.2416687, 135.4099579
26: -101.2567291, 82.4935608, -101.1954575, 82.2693787, -183.5261078, 183.6890106
27: -88.7816315, 60.0362549, -88.7280045, 59.8535423, -148.6351776, 148.7642517
28: -71.7849121, 61.1729202, -71.7509613, 60.9505463, -132.7354584, 132.9238892
29: -92.4243851, 48.8540039, -92.3502579, 48.6808472, -141.1052246, 141.2042542
30: -88.5536041, 65.5886536, -88.4967728, 65.4658966, -154.0195007, 154.0854187
31: -93.4072876, 55.3718300, -93.3460083, 55.1334686, -148.5407562, 148.7178345
32: -93.2344971, 59.3088646, -93.1821671, 59.1786842, -152.4131775, 152.4910278
33: -121.7455750, 78.9248047, -121.6677933, 78.7726746, -200.5182343, 200.5925903
34: -100.7999954, 58.1814842, -100.7448196, 57.9731216, -158.7731171, 158.9263000
35: -98.9544449, 60.8424149, -98.8985519, 60.6291008, -159.5835419, 159.7409515
36: -100.5111084, 63.0918503, -100.4636993, 62.8268776, -163.3379669, 163.5555420
37: -141.7966919, 62.0811195, -141.7026825, 61.9940338, -203.7907104, 203.7837982
38: -119.7850189, 78.9003143, -119.7178802, 78.6262894, -198.4113159, 198.6181946
39: -135.1227722, 75.6557083, -135.0173340, 75.5386658, -210.6614380, 210.6730347
40: -112.0686951, 61.3472137, -111.9836273, 61.2663536, -173.3350525, 173.3308258
41: -94.2443008, 64.7332458, -94.1823502, 64.6313782, -158.8756714, 158.9155884
42: -69.2570343, 60.0888290, -69.2140427, 60.0024147, -129.2594299, 129.3028717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4690751
time: 224.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4733102
time: 128.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 355.75 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4074596
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4121865
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4247705
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4290534
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4286816
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4333430
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4690751
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 355.75
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4733102

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -105.5659256, 78.1810226, -105.5836716, 78.3173676, -183.8833008, 183.7646942
1: -58.6664124, 65.4096375, -58.6753159, 65.5173569, -124.1837692, 124.0849533
2: -52.5511818, 63.0073357, -52.5553398, 63.1113205, -115.6624908, 115.5626755
3: -57.9298706, 72.4026566, -57.9366608, 72.4917145, -130.4215851, 130.3393250
4: -66.1084976, 73.6292419, -66.0971680, 73.7390137, -139.8475037, 139.7263947
5: -59.9828491, 70.2068481, -59.9887848, 70.3011627, -130.2840118, 130.1956329
6: -95.1651154, 62.1137772, -95.2457657, 62.1497383, -157.3148499, 157.3595276
7: -70.1066666, 67.1701736, -70.1233521, 67.2993317, -137.4059906, 137.2935181
8: -76.3486633, 89.6261215, -76.3755951, 89.7875824, -166.1362305, 166.0017090
9: -64.3032379, 70.9278564, -64.3089218, 71.0430450, -135.3462830, 135.2367554
10: -94.8847656, 90.4943619, -94.9111404, 90.5859909, -185.4707184, 185.4055023
11: -91.5472183, 55.7893562, -91.5700531, 55.7748871, -147.3221130, 147.3594055
12: -90.7518158, 72.2503586, -90.8326340, 72.3288345, -163.0806427, 163.0829773
13: -96.5488892, 95.3520813, -96.5195084, 95.3843613, -191.9332123, 191.8715820
14: -141.1407623, 79.8846893, -141.1531982, 79.9547882, -221.0955505, 221.0378723
15: -75.0346527, 67.0635071, -75.0625153, 67.1023102, -142.1369629, 142.1260223
16: -95.5124283, 69.2307434, -95.5374527, 69.3037262, -164.8161621, 164.7681885
17: -138.6927948, 76.0640717, -138.6887817, 76.0958710, -214.7886658, 214.7528534
18: -90.6196442, 74.9764023, -90.7075195, 75.0047989, -165.6244507, 165.6839294
19: -72.4477997, 45.2290039, -72.5254364, 45.2063179, -117.6541061, 117.7544174
20: -67.3546600, 52.2057571, -67.4262848, 52.2055397, -119.5601959, 119.6320343
21: -88.7199097, 53.1960526, -88.8023376, 53.1815033, -141.9014130, 141.9983826
22: -90.4938812, 54.6614532, -90.6174393, 54.6621132, -145.1559906, 145.2789001
23: -71.1229248, 55.3632507, -71.1733246, 55.3288345, -126.4517517, 126.5365753
24: -85.7279587, 50.9910507, -85.8001404, 50.9657173, -136.6936798, 136.7911987
25: -75.8677673, 58.5377617, -75.9851074, 58.5339584, -134.4017181, 134.5228577
26: -100.7621994, 81.6953125, -100.8466644, 81.6956482, -182.4578552, 182.5419617
27: -88.3079834, 59.4566879, -88.3969193, 59.4272156, -147.7351837, 147.8536072
28: -71.3831177, 60.4842415, -71.4696884, 60.4592133, -131.8423309, 131.9539185
29: -91.9117355, 48.3224640, -92.0114594, 48.3003235, -140.2120667, 140.3339233
30: -88.2014084, 65.1792374, -88.2502823, 65.1796265, -153.3810272, 153.4295197
31: -92.8847122, 54.6587410, -93.0141907, 54.6556015, -147.5403137, 147.6729279
32: -92.8603821, 58.7914581, -92.9272614, 58.8321037, -151.6924896, 151.7187042
33: -121.2825470, 78.2724762, -121.3595352, 78.3217392, -199.6042786, 199.6320190
34: -100.4107971, 57.4516792, -100.4971237, 57.4763794, -157.8871765, 157.9487915
35: -98.5152359, 60.1216011, -98.6223755, 60.1612473, -158.6764832, 158.7439728
36: -100.0529633, 62.2461662, -100.1833344, 62.2862358, -162.3392029, 162.4295044
37: -141.2580566, 61.5686569, -141.3447876, 61.6087341, -202.8667603, 202.9134521
38: -119.2109451, 77.9812927, -119.3641357, 78.0521851, -197.2631226, 197.3454285
39: -134.6157684, 75.2037964, -134.6874390, 75.2717056, -209.8874817, 209.8912201
40: -111.6591339, 60.9089165, -111.7224884, 60.9688377, -172.6279602, 172.6314087
41: -93.8790054, 64.2189255, -93.9212189, 64.2133255, -158.0923157, 158.1401367
42: -69.0269470, 59.6108894, -69.0498962, 59.6180611, -128.6449890, 128.6607819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4052395
time: 140.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4052395
time: 146.83 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -106.1426849, 78.3680420, -105.7842331, 78.3330765, -184.4757690, 184.1522827
1: -59.0355530, 65.5451279, -58.8054848, 65.5267334, -124.5622864, 124.3506088
2: -52.8619156, 63.1337662, -52.6655769, 63.1225014, -115.9844055, 115.7993393
3: -58.2218666, 72.5419617, -58.0403481, 72.5090942, -130.7309570, 130.5823059
4: -66.4962463, 73.7700424, -66.2336273, 73.7521133, -140.2483521, 140.0036621
5: -60.3001862, 70.3602753, -60.1013565, 70.3179550, -130.6181335, 130.4616241
6: -95.3017654, 62.4159698, -95.2673111, 62.2547493, -157.5565186, 157.6832886
7: -70.5276031, 67.3322372, -70.2736664, 67.3129425, -137.8405457, 137.6058960
8: -76.8065872, 89.8125992, -76.5376434, 89.8068619, -166.6134491, 166.3502350
9: -64.6686020, 71.0857697, -64.4367523, 71.0591736, -135.7277832, 135.5225220
10: -95.2827835, 90.6590271, -95.0490952, 90.6132050, -185.8959808, 185.7080994
11: -91.6785507, 55.9628181, -91.6099472, 55.8339729, -147.5125275, 147.5727539
12: -90.8987885, 72.6046906, -90.8560791, 72.4495850, -163.3483582, 163.4607697
13: -96.8828049, 95.4987335, -96.6343536, 95.4173584, -192.3001709, 192.1330566
14: -141.6001892, 80.0155029, -141.3101807, 79.9728622, -221.5730133, 221.3256836
15: -75.2889557, 67.1206512, -75.1471252, 67.1171265, -142.4060822, 142.2677612
16: -95.9089050, 69.3803864, -95.6754456, 69.3249664, -165.2338715, 165.0558319
17: -139.1074524, 76.1700745, -138.8301697, 76.1224060, -215.2298431, 215.0002289
18: -90.7841797, 75.2700806, -90.7393417, 75.1076508, -165.8918152, 166.0094299
19: -72.5776825, 45.4682159, -72.5494232, 45.2922668, -117.8699493, 118.0176392
20: -67.4668121, 52.4427414, -67.4472351, 52.2888451, -119.7556610, 119.8899765
21: -88.8781052, 53.4246826, -88.8381577, 53.2618713, -142.1399689, 142.2628479
22: -90.6743240, 54.9320641, -90.6514587, 54.7585907, -145.4329224, 145.5835266
23: -71.2416306, 55.5976562, -71.1948090, 55.4108696, -126.6524963, 126.7924652
24: -85.8595428, 51.1982574, -85.8293533, 51.0385056, -136.8980408, 137.0275879
25: -76.0154190, 58.8050194, -76.0108109, 58.6280861, -134.6435089, 134.8158264
26: -100.9651337, 82.0777283, -100.8794937, 81.8314590, -182.7966003, 182.9572144
27: -88.4625549, 59.7118835, -88.4247513, 59.5159225, -147.9784851, 148.1366272
28: -71.5190430, 60.7901230, -71.4881134, 60.5667458, -132.0857849, 132.2782135
29: -92.0846100, 48.5538139, -92.0491791, 48.3821259, -140.4667358, 140.6029968
30: -88.3114548, 65.3477325, -88.2815933, 65.2358856, -153.5473328, 153.6293335
31: -93.0635529, 54.9785995, -93.0452957, 54.7691803, -147.8327332, 148.0238800
32: -93.0021515, 59.0558357, -92.9513321, 58.9238052, -151.9259644, 152.0071716
33: -121.4786301, 78.6118011, -121.3984528, 78.4397278, -199.9183655, 200.0102539
34: -100.5688171, 57.8057175, -100.5231094, 57.6021194, -158.1709290, 158.3288269
35: -98.6939316, 60.4763718, -98.6515045, 60.2871552, -158.9810791, 159.1278687
36: -100.2362442, 62.6581039, -100.2062225, 62.4330750, -162.6693115, 162.8643188
37: -141.4891968, 61.8685455, -141.3883972, 61.7149506, -203.2041473, 203.2569275
38: -119.4515152, 78.4305573, -119.3975143, 78.2107620, -197.6622620, 197.8280640
39: -134.8169098, 75.4462585, -134.7355042, 75.3573151, -210.1742249, 210.1817322
40: -111.8269958, 61.1658478, -111.7588348, 61.0580673, -172.8850555, 172.9246826
41: -94.0284576, 64.4912338, -93.9481735, 64.3086700, -158.3371277, 158.4394073
42: -69.1225433, 59.8610878, -69.0691986, 59.7050781, -128.8276215, 128.9302826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4250209, upper bound: 79.4077165
time: 415.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4077165
time: 107.76 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -105.7920685, 78.2022400, -106.2293777, 78.5540848, -184.3461456, 184.4316101
1: -58.8061066, 65.4222565, -59.0731010, 65.6793442, -124.4854507, 124.4953613
2: -52.6707764, 63.0211792, -52.8923149, 63.2578392, -115.9286118, 115.9134979
3: -58.0391502, 72.4232635, -58.2466202, 72.6619034, -130.7010498, 130.6698914
4: -66.2655411, 73.6453247, -66.5452881, 73.9136810, -140.1791992, 140.1906128
5: -60.1026382, 70.2285156, -60.3300285, 70.5003052, -130.6029358, 130.5585480
6: -95.1947098, 62.2193489, -95.4022522, 62.4490623, -157.6437683, 157.6215973
7: -70.2617493, 67.1854095, -70.5614243, 67.4837265, -137.7454834, 137.7468262
8: -76.5109711, 89.6481628, -76.8406219, 89.9865952, -166.4975586, 166.4887848
9: -64.4449921, 70.9463348, -64.7183762, 71.2287445, -135.6736908, 135.6647034
10: -95.0291367, 90.5263290, -95.3334274, 90.7806778, -185.8097839, 185.8597565
11: -91.5999985, 55.8714218, -91.7724380, 56.0104599, -147.6104584, 147.6438446
12: -90.7795715, 72.3527374, -90.9846420, 72.6298828, -163.4094238, 163.3373718
13: -96.7014923, 95.3930359, -96.9594727, 95.6343231, -192.3358154, 192.3525085
14: -141.3276978, 79.9067535, -141.6954346, 80.1235352, -221.4512329, 221.6021729
15: -75.1247864, 67.0869598, -75.3331757, 67.1927338, -142.3175201, 142.4201355
16: -95.6668396, 69.2544403, -95.9846420, 69.4892426, -165.1560822, 165.2390594
17: -138.8684082, 76.0953522, -139.1968079, 76.2481384, -215.1165466, 215.2921600
18: -90.6569672, 75.0751648, -90.8946228, 75.2963486, -165.9533081, 165.9697876
19: -72.4773407, 45.3342667, -72.7046661, 45.4993172, -117.9766541, 118.0389328
20: -67.3816986, 52.3032455, -67.5708160, 52.4798813, -119.8615723, 119.8740616
21: -88.7636032, 53.2945518, -89.0102539, 53.4603920, -142.2239990, 142.3048096
22: -90.5336456, 54.7680130, -90.8448181, 54.9631157, -145.4967651, 145.6128235
23: -71.1500092, 55.4743576, -71.3565750, 55.6449852, -126.7949982, 126.8309174
24: -85.7666016, 51.0852585, -85.9953156, 51.2344513, -137.0010529, 137.0805664
25: -75.8981018, 58.6458473, -76.1614532, 58.8399315, -134.7380371, 134.8072968
26: -100.8057861, 81.8457489, -101.1052628, 82.1223602, -182.9281464, 182.9510193
27: -88.3435745, 59.5717926, -88.6292343, 59.7565651, -148.1001434, 148.2010193
28: -71.4071045, 60.6170425, -71.6684952, 60.8343277, -132.2414246, 132.2855377
29: -91.9558182, 48.4244232, -92.2452698, 48.5907555, -140.5465393, 140.6696777
30: -88.2422562, 65.2518768, -88.4209137, 65.3962097, -153.6384583, 153.6727905
31: -92.9214478, 54.7843132, -93.2388382, 55.0105476, -147.9319916, 148.0231476
32: -92.8937607, 58.8775940, -93.1127701, 59.0749130, -151.9686584, 151.9903564
33: -121.3331604, 78.3791580, -121.5865173, 78.6389771, -199.9721222, 199.9656677
34: -100.4431992, 57.5796165, -100.6754837, 57.8365364, -158.2797394, 158.2550964
35: -98.5519943, 60.2368927, -98.8178711, 60.4926033, -159.0446014, 159.0547485
36: -100.0812378, 62.3816147, -100.3788147, 62.6706467, -162.7518768, 162.7604218
37: -141.3131104, 61.6623917, -141.6116638, 61.8783569, -203.1914368, 203.2740479
38: -119.2499161, 78.1201553, -119.6151199, 78.4513245, -197.7012329, 197.7352753
39: -134.6827698, 75.2602234, -134.9295197, 75.4425201, -210.1252594, 210.1897278
40: -111.7066956, 60.9798584, -111.9149017, 61.1694260, -172.8760986, 172.8947449
41: -93.9162979, 64.3315353, -94.1203613, 64.5279236, -158.4442139, 158.4518738
42: -69.0532227, 59.7121964, -69.1757050, 59.9050522, -128.9582672, 128.8878784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4223701
time: 123.25 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4223701
time: 104.97 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -106.3692398, 78.3892288, -106.4308548, 78.5698318, -184.9390717, 184.8200836
1: -59.1753693, 65.5577240, -59.2034912, 65.6887283, -124.8640976, 124.7611923
2: -52.9817467, 63.1476593, -53.0028915, 63.2690544, -116.2507858, 116.1505508
3: -58.3314667, 72.5625381, -58.3506851, 72.6794357, -131.0108948, 130.9132080
4: -66.6537704, 73.7862244, -66.6823883, 73.9268570, -140.5806274, 140.4686127
5: -60.4202423, 70.3818970, -60.4429703, 70.5171280, -130.9373474, 130.8248596
6: -95.3313751, 62.5217972, -95.4240494, 62.5544472, -157.8858185, 157.9458313
7: -70.6828384, 67.3474045, -70.7119751, 67.4973450, -138.1801758, 138.0593872
8: -76.9695129, 89.8346558, -77.0035706, 90.0059967, -166.9755096, 166.8382263
9: -64.8106918, 71.1042328, -64.8466797, 71.2449493, -136.0556335, 135.9508972
10: -95.4278564, 90.6910858, -95.4720306, 90.8079376, -186.2357788, 186.1631165
11: -91.7310410, 56.0455055, -91.8118744, 56.0702362, -147.8012695, 147.8573761
12: -90.9264984, 72.7077713, -91.0080032, 72.7517395, -163.6782227, 163.7157745
13: -97.0360718, 95.5395355, -97.0750351, 95.6673965, -192.7034607, 192.6145630
14: -141.7873535, 80.0374298, -141.8531952, 80.1415253, -221.9288635, 221.8906250
15: -75.3795624, 67.1442184, -75.4183044, 67.2077179, -142.5872803, 142.5625305
16: -96.0635834, 69.4040222, -96.1228256, 69.5104370, -165.5740204, 165.5268555
17: -139.2821350, 76.2015076, -139.3377991, 76.2745590, -215.5567017, 215.5393066
18: -90.8215332, 75.3696747, -90.9263763, 75.4000397, -166.2215576, 166.2960358
19: -72.6071930, 45.5740776, -72.7285767, 45.5859070, -118.1931000, 118.3026505
20: -67.4939346, 52.5405312, -67.5918808, 52.5637703, -120.0577011, 120.1324158
21: -88.9219208, 53.5238838, -89.0459671, 53.5414581, -142.4633789, 142.5698547
22: -90.7139206, 55.0391235, -90.8785934, 55.0602608, -145.7741852, 145.9177094
23: -71.2686920, 55.7094765, -71.3780975, 55.7277184, -126.9963989, 127.0875702
24: -85.8982697, 51.2933502, -86.0249176, 51.3080788, -137.2063446, 137.3182678
25: -76.0458374, 58.9137917, -76.1873169, 58.9349098, -134.9807434, 135.1011047
26: -101.0087204, 82.2286301, -101.1381073, 82.2588806, -183.2675781, 183.3667145
27: -88.4983826, 59.8279457, -88.6573105, 59.8461761, -148.3445587, 148.4852600
28: -71.5431290, 60.9236031, -71.6869812, 60.9426918, -132.4858093, 132.6105652
29: -92.1285095, 48.6562767, -92.2826996, 48.6731415, -140.8016510, 140.9389648
30: -88.3524933, 65.4214249, -88.4523239, 65.4534149, -153.8059082, 153.8737335
31: -93.1002502, 55.1048470, -93.2702484, 55.1249313, -148.2251892, 148.3750763
32: -93.0355301, 59.1424408, -93.1369247, 59.1671143, -152.2026367, 152.2793579
33: -121.5290756, 78.7191620, -121.6254578, 78.7576675, -200.2867432, 200.3446198
34: -100.6012497, 57.9337540, -100.7015762, 57.9622955, -158.5635376, 158.6353302
35: -98.7306595, 60.5919571, -98.8471146, 60.6188278, -159.3494873, 159.4390717
36: -100.2644272, 62.7939034, -100.4015961, 62.8178482, -163.0822754, 163.1954956
37: -141.5440216, 61.9628181, -141.6554260, 61.9852257, -203.5292358, 203.6182251
38: -119.4904099, 78.5696564, -119.6485062, 78.6102982, -198.1007080, 198.2181702
39: -134.8834229, 75.5028687, -134.9775696, 75.5280914, -210.4114990, 210.4804230
40: -111.8744583, 61.2363701, -111.9511032, 61.2578278, -173.1322937, 173.1874695
41: -94.0656586, 64.6040802, -94.1474380, 64.6237488, -158.6893921, 158.7515259
42: -69.1489639, 59.9626122, -69.1949234, 59.9929581, -129.1419220, 129.1575317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4250610
time: 112.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4707022, upper bound: 79.4250610
time: 110.37 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -105.9862061, 78.4636841, -105.6050110, 78.3973846, -184.3835754, 184.0686646
1: -58.9524841, 65.6145325, -58.6851463, 65.5758057, -124.5282898, 124.2996674
2: -52.7817383, 63.2031479, -52.5623741, 63.1654243, -115.9471588, 115.7655182
3: -58.1311798, 72.5992661, -57.9430962, 72.5432816, -130.6744690, 130.5423584
4: -66.3820190, 73.8525085, -66.1070251, 73.8011017, -140.1831055, 139.9595337
5: -60.1765251, 70.4327316, -59.9949188, 70.3610382, -130.5375671, 130.4276428
6: -95.3422241, 62.2799683, -95.2865829, 62.1620865, -157.5043030, 157.5665436
7: -70.4038696, 67.4049530, -70.1316605, 67.3653946, -137.7692413, 137.5366211
8: -76.6940765, 89.9075165, -76.3874817, 89.8635941, -166.5576477, 166.2949982
9: -64.5656281, 71.1551514, -64.3219452, 71.1053696, -135.6709900, 135.4770813
10: -95.1825104, 90.7045135, -94.9272308, 90.6373749, -185.8198853, 185.6317444
11: -91.7563324, 55.9245605, -91.6033020, 55.7841949, -147.5405273, 147.5278625
12: -90.9116211, 72.4406738, -90.8683472, 72.3460617, -163.2576904, 163.3090210
13: -96.7863922, 95.5939255, -96.5330887, 95.4403610, -192.2267456, 192.1270142
14: -141.4429932, 80.0683289, -141.1751099, 79.9987793, -221.4417419, 221.2434387
15: -75.2322693, 67.1893616, -75.0782318, 67.1286926, -142.3609619, 142.2675934
16: -95.8474960, 69.4152222, -95.5591202, 69.3507462, -165.1982269, 164.9743347
17: -139.0259399, 76.2108765, -138.7072144, 76.1232529, -215.1492004, 214.9180908
18: -90.8312149, 75.1939621, -90.7538376, 75.0156403, -165.8468475, 165.9477997
19: -72.6697845, 45.4144897, -72.5785980, 45.2104492, -117.8802338, 117.9930878
20: -67.5288849, 52.3927422, -67.4692078, 52.2133102, -119.7421951, 119.8619385
21: -88.9673080, 53.3765259, -88.8566284, 53.1881294, -142.1554413, 142.2331543
22: -90.7895279, 54.8819809, -90.6892471, 54.6701698, -145.4597015, 145.5712280
23: -71.3221817, 55.5337677, -71.2215881, 55.3358231, -126.6580048, 126.7553558
24: -85.9592209, 51.1568794, -85.8562927, 50.9712372, -136.9304504, 137.0131683
25: -76.1152115, 58.7719650, -76.0476837, 58.5418892, -134.6571045, 134.8196411
26: -101.0050430, 81.9461517, -100.9041061, 81.7043304, -182.7093506, 182.8502502
27: -88.5874634, 59.6556282, -88.4671478, 59.4344139, -148.0218658, 148.1227722
28: -71.6217575, 60.7222366, -71.5334778, 60.4662247, -132.0879822, 132.2557068
29: -92.2029572, 48.5112762, -92.0780182, 48.3070679, -140.5100098, 140.5892944
30: -88.3966675, 65.3387909, -88.2936554, 65.1920929, -153.5887604, 153.6324463
31: -93.1880264, 54.9126091, -93.0894241, 54.6626892, -147.8507080, 148.0020294
32: -93.0562820, 58.9456482, -92.9722824, 58.8424606, -151.8987274, 151.9179230
33: -121.4948883, 78.4653473, -121.4013672, 78.3345413, -199.8294373, 199.8666992
34: -100.6063995, 57.6877022, -100.5425339, 57.4848099, -158.0912018, 158.2302399
35: -98.7350922, 60.3594856, -98.6737366, 60.1693726, -158.9044495, 159.0332184
36: -100.2965164, 62.5288620, -100.2456207, 62.2927361, -162.5892487, 162.7744751
37: -141.5048676, 61.6713829, -141.3927002, 61.6155396, -203.1204071, 203.0640869
38: -119.5013580, 78.2964783, -119.4364471, 78.0652924, -197.5666504, 197.7329102
39: -134.8498230, 75.3456726, -134.7262573, 75.2804947, -210.1303101, 210.0719299
40: -111.8491211, 61.0125427, -111.7534714, 60.9768372, -172.8259583, 172.7660217
41: -94.0548248, 64.3381500, -93.9569550, 64.2204895, -158.2753143, 158.2951050
42: -69.1330414, 59.7243690, -69.0684509, 59.6276512, -128.7606964, 128.7928162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4269366
time: 112.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4269366
time: 188.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -106.5606766, 78.6501846, -105.8052979, 78.4130630, -184.9737244, 184.4554749
1: -59.3199081, 65.7497711, -58.8151894, 65.5851364, -124.9050446, 124.5649567
2: -53.0912971, 63.3293152, -52.6724892, 63.1765747, -116.2678680, 116.0018005
3: -58.4219666, 72.7383118, -58.0466843, 72.5606308, -130.9825745, 130.7849731
4: -66.7684326, 73.9927063, -66.2433395, 73.8141632, -140.5825958, 140.2360535
5: -60.4926147, 70.5856934, -60.1073799, 70.3778229, -130.8704376, 130.6930695
6: -95.4773941, 62.5813484, -95.3080368, 62.2669334, -157.7443237, 157.8893890
7: -70.8231506, 67.5665741, -70.2818375, 67.3790131, -138.2021637, 137.8484192
8: -77.1501465, 90.0935364, -76.5493927, 89.8829041, -167.0330505, 166.6429291
9: -64.9300613, 71.3123627, -64.4496536, 71.1214523, -136.0515137, 135.7620239
10: -95.5787659, 90.8681412, -95.0650253, 90.6645584, -186.2432861, 185.9331360
11: -91.8856888, 56.0974388, -91.6430588, 55.8431320, -147.7288208, 147.7404938
12: -91.0573120, 72.7938690, -90.8917084, 72.4667053, -163.5240173, 163.6855774
13: -97.1187363, 95.7389450, -96.6477203, 95.4732895, -192.5920258, 192.3866577
14: -141.8999176, 80.1980057, -141.3318481, 80.0167694, -221.9166870, 221.5298462
15: -75.4857330, 67.2458420, -75.1627197, 67.1434326, -142.6291656, 142.4085693
16: -96.2422333, 69.5641098, -95.6970215, 69.3719101, -165.6141357, 165.2611389
17: -139.4370422, 76.3159561, -138.8485107, 76.1497269, -215.5867615, 215.1644592
18: -90.9949265, 75.4859772, -90.7855911, 75.1183472, -166.1132660, 166.2715454
19: -72.7991943, 45.6518326, -72.6025696, 45.2962570, -118.0954437, 118.2543945
20: -67.6403427, 52.6283035, -67.4901276, 52.2964859, -119.9368286, 120.1184158
21: -89.1242981, 53.6032677, -88.8923416, 53.2683640, -142.3926697, 142.4956055
22: -90.9688568, 55.1508713, -90.7232437, 54.7665367, -145.7353973, 145.8741150
23: -71.4402084, 55.7665405, -71.2430267, 55.4177094, -126.8579102, 127.0095520
24: -86.0900955, 51.3627548, -85.8854446, 51.0438919, -137.1339874, 137.2481995
25: -76.2622223, 59.0373955, -76.0733337, 58.6359215, -134.8981476, 135.1107178
26: -101.2075272, 82.3263626, -100.9368973, 81.8399963, -183.0475159, 183.2632599
27: -88.7408142, 59.9091682, -88.4948883, 59.5229530, -148.2637634, 148.4040527
28: -71.7570648, 61.0260429, -71.5518570, 60.5736008, -132.3306580, 132.5778961
29: -92.3741989, 48.7412720, -92.1156693, 48.3887558, -140.7629547, 140.8569336
30: -88.5053406, 65.5061646, -88.3248901, 65.2482147, -153.7535553, 153.8310547
31: -93.3661575, 55.2304344, -93.1204681, 54.7760849, -148.1422424, 148.3509064
32: -93.1966553, 59.2094002, -92.9963150, 58.9340630, -152.1307220, 152.2056885
33: -121.6897888, 78.8032532, -121.4402313, 78.4524689, -200.1422577, 200.2434845
34: -100.7630386, 58.0404015, -100.5684433, 57.6104431, -158.3734741, 158.6088409
35: -98.9128723, 60.7125511, -98.7028198, 60.2952271, -159.2080994, 159.4153748
36: -100.4793015, 62.9396248, -100.2684631, 62.4394569, -162.9187622, 163.2080841
37: -141.7343445, 61.9702606, -141.4362335, 61.7217178, -203.4560242, 203.4064789
38: -119.7410660, 78.7441711, -119.4697647, 78.2238312, -197.9648743, 198.2139282
39: -135.0496063, 75.5867004, -134.7743225, 75.3660660, -210.4156799, 210.3610229
40: -112.0151062, 61.2678375, -111.7897644, 61.0659790, -173.0810852, 173.0576019
41: -94.2027206, 64.6093216, -93.9838257, 64.3156281, -158.5183411, 158.5931396
42: -69.2269745, 59.9728050, -69.0876694, 59.7144699, -128.9414368, 129.0604706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 829

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4250209, upper bound: 79.4300770
time: 123.22 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4300770
time: 150.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 276.60 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4052395
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4052395
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4250209, upper bound: 79.4077165
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4077165
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4223701
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4223701
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4250610
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4707022, upper bound: 79.4250610
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4269366
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4269366
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4250209, upper bound: 79.4300770
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 276.60
Output dim: 1, lower bound: -79.4036823, upper bound: 79.4300770
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 276.60
Output dim: 1, lower bound: -79.4241150, upper bound: 79.4690751
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 276.60
Output dim: 1, lower bound: -79.4733102, upper bound: 79.4733102
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=124.9945068359375
rel_dist={1: [-79.47797671394284, 79.47797670615168]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 14154.20 seconds
