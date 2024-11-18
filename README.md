# jaxLearning
First step:

  - Learn jax and flax basics ✔
  
[second steps](./flax/readme.md):

  - Develop a simple flax model ✔
  
  - Save and load ✔
  
    * There is an issue with saving drop out related to rng dtype on orbax (change type before save and revert after load)
  
  - Save based on eval dataset
    * Early stopping
  - Custom metrics (a somewhat functional metric is in [riceTypes.ipynp](./flax/riceTypes.ipynb))
      * the existing accuracy is argmax !!!  
    
  - parallelize it with pmap
