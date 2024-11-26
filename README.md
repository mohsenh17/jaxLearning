# jaxLearning
First step:

  - Learn jax and flax basics ✔
  
[second steps](./flax/readme.md) ([flaxCheckpoint.ipynb](./flax/flaxCheckpoint.ipynb)):

  - Develop a simple flax model ✔
  
  - Save and load ✔
  
    * There is an issue with saving drop out related to rng dtype on orbax (change type before save and revert after load)  ✔
  
  - Save based on eval dataset ✔
    * Early stopping ✔
  - Custom metrics f1 score, precision, recall ([CustomMetrics.ipynb](./flax/CustomMetrics.ipynb)) ✔
      * the existing accuracy is argmax !!! -> a custom accuracy added  ✔
    
  - parallelization
      * A simple MLP parallel (data:2, model:4) with sharding ([parallelMLP.ipynb](./flax/parallelMLP.ipynb)) ✔
