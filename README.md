# SOAT
This is the code for the paper [SOAT: A Scenario-Oriented Adversarial Training Approach towards Universal Control].

In the field of robotic control, universal control policies model robot morphologies as graph structures and utilize Graph Neural Networks (GNNs) and Transformers to process variable-length observations and actions, enabling generalized control across diverse robot morphologies. However, current algorithms are often trained and evaluated with highly similar morphologies, restricting their utility in downstream tasks that require generalised control ability, such as morphological design optimization and robust control against significant structural damage. To overcome this issue, we introduce SOAT, a novel pre-training method that constructs an adversarial set of morphology samples tailored to the specific requirements of downstream scenarios, enhancing the performance of universal control policies. Experimental results in scenarios involving damaged robot morphologies demonstrate that SOAT significantly improves the generalization capabilities of these policies, offering a robust solution for diverse and evolving robotic systems.


### Requirements
* Python 3.7
* Linux
  
```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

Install Python dependencies with pip:

```shell
pip install -r requirements.txt
```

## Running the code
For multi-robot training, run the following commands to train. 

```shell
python examples/run_soat.py --env-name "Walker-v0" --algo ppo --use-gae --lr 0.0001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 16 --num-steps 128 --num-mini-batch 12 --log-interva  100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50
```

## Hyperparameter

The meaning and values of the hyperparameters required for experimental running can be found in examples/ppo/arguments.py

