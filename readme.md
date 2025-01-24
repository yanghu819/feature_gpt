Making language model generation continuous by evolving in feature space.

Past tokens:    [1, 2, 3]
Future tokens:  [2, 3, 4]
Current features:  F[1,2,3]
Target features:   F[2,3,4]
Time t=0.5:  F[1,2,3] ──> F[1.5, 2.5, 3.5] ──> F[2,3,4]
Core idea: Predict how features change between states



