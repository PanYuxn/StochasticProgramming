\ Model First_Stage_Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  10 x1 + theta2
Subject To
 optimality_cut_0: 15 x1 + theta2 >= 1807.5
 optimality_cut_1: 15 x1 + theta2 >= 2307.5
 feasibility_cut_0: x1 <= 120
 feasibility_cut_1: x1 <= 100
Bounds
End
