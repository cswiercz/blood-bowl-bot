# blood-bowl-bot

Using Blood Bowl to teach myself reinforcement learning. See http://cswiercz.info.

```bash
$ docker build . -t bbb:latest
$ docker run -it bbb --training_episodes 10 --evaluation_episodes 1
```

## Running Cart-Pole Locally

See the Dockerfile for requirements. Once you have those installed, you can run
cart-pole without Docker like so,

```bash
$ python3 run_cart_pole.py --training_episodes 300 --evaluation_episodes 10 --with_render
```

For a full list of command line options run

```bash
$ python3 run_cart_pole.py --help
```

Currently, the choice of model is hard-coded into `run_cart_pole.py`. Edit the
file to test different models. Use the following for producing the plots in the
`log-parser` notebook,

```bash
$ for n in {0..6}; do python3 run_cart_pole.py --training_episodes 1000 --evaluation_episodes 0 --logfile ./logs/cart-pole-dueling-dqn-$n.log ; done;
```
