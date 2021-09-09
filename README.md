# sepsis-examples
This repository contains all the code necessary to replicate the results of the paper "Methods For Off-policy Evaluation Under Unobserved Confounding", where the simulator used for generating observations, packages for importance sampling estimation and bounds computation, and the code for design sensitivity analysis in the first example are written based on code provided by [Namkoong, Keramati, Yadlowsky, Brunskill](https://github.com/StanfordAI4HI/off_policy_confounding/tree/master/sepsis), and the simulator used for generating observations and the codes for constructing MDPs, producing behaviour and evaluation policies, and counterfactual evaluation in the second example are coded based on work of [Oberst, Sontag](https://github.com/clinicalml/gumbel-max-scm).

One can obtain data needed for reproducing the first example by unzipping ``processed.zip`` provided by [Namkoong, Keramati, Yadlowsky, Brunskill](https://github.com/StanfordAI4HI/off_policy_confounding/tree/master/sepsis). And all data needed for the second example is provided by [Oberst, Sontag](https://github.com/clinicalml/gumbel-max-scm), where one can access it by unzipping ``diab_txr_mats-replication.zip``.
