class Agent:
    """Base class for RL agents acting on OpenAI gym environments.

    The three primary functions of an agent are (1) acting on an environment,
    (2) storing state-action-reward information to memory, and (3) updating its
    model.
    """

    def act(self, environment, **kwds):
        """Act on an environment.

        Parameters
        ----------
        environment
            An OpenAI gym environment.

        Returns
        -------
        action
            An action interpretable by the environment.
        """
        raise NotImplementedError()


    def remember(self, state, action, reward, next_state, is_terminal):
        """Store environment information to the agent's memory.

        Parameters
        ----------
        state
            An environment state.
        action
            The action taken by this agent based on the state.
        reward
            Reward received for taking this action.
        next_state
            The resulting state after taking the action.
        is_terminal
            If `True`, indicates that resulting next state is the final state
            of the episode.
        """
        pass


    def update(self, **kwds):
        """Update the agent's model.

        Returns
        -------
        None
        """
        pass


    def finalize(self, *args, **kwds):
        """Perform any post-training actions for final model evaluation."""
        pass
