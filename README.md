# Hybrid MARL - Neurips submission #6996

Hybrid MARL is an extension over the EPyMARL library.


# Installation instructions

To install the codebase, run (tested with python 3.8.10): 
```sh
./install.sh
```

The MPE and LBF environments are already included in the provided folder. 

## Running experiments
After installation, you can use the script run.sh to run experiments, where:
1) `ENV` variable selects the environment to use:
    - `SimpleSpreadXY-v0` - SpreadXY-2
    - `SimpleSpreadXY4-v0` - SpreadXY-4;
    - `SimpleSpreadBlind-v0` - SpreadBlindfold;
    - `SimpleBlindDeaf-v0` - HearSee;
    - `SimpleSpread-v0` - SimpleSpread;
    - `SimpleSpeakerListener-v0` - SimpleSpeakerListener;
    - `Foraging-2s-15x15-2p-2f-coop-v2` - LBF environment (modified);
2) `ALGO` variable selects the RL algorithm to use (`iql_ns`, `qmix_ns`, `ippo_ns`, or `mappo_ns` for MPE environments; `iql_ns_lbf`, `qmix_ns_lbf`, `ippo_ns_lbf`, or `mappo_ns_lbf` for LBF environments).
3)  `PERCEPTION` variable selects the perceptual model to use:
    - `obs` - Obs.
    - `joint_obs` - Oracle;
    - `joint_obs_drop_test` - Masked joint obs.;
    - `ablation_no_pred` - MD baseline;
    - `ablation_no_pred_masks` - MD w/ masks baseline;
    - `maro_no_training` - MARO;
    - `maro` - MARO w/ dropout;
    - `SimpleBlindDeaf-v0` - HearSee;
4) `TIME_LIMIT` variable: 25 for MPE environments; 30 for LBF environments.
   
