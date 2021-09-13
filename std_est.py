"""This file containts std_est class that is used to
estimate importance sampling and per-decision importance 
sampling of unconfounded MDP. The main part of the code is borrowed 
from https://github.com/StanfordAI4HI/off_policy_confounding/blob/master/sepsis/core/conf_wis.py
"""
import numpy as np
from tqdm import tqdm

class std_est(object):
    def __init__(self, trajectories, returns, config, iter_ids, iter_ius, k):
        """Computing importance sampling or per-decision importance sampling estimate
        for a unconfounded matrix MDP. This class assumes there 
        are two policies acting one before time step k, and one after.
        This class assumes tabular representation of state, action
        Parameters
        ----------
        trajectories : np.array, float [None, max_horizon, 5]
            [None, max_horizon, 0] : timestep
            [None, max_horizon, 1] : action taken, -1 default
            [None, max_horizon, 2] : state index
            [None, max_horizon, 3] : next state index
            [None, max_horizon, 4] : reward
        returns : np.array, float [None] for compute()
                            float [None, max_horizon] for compute_pd()
            discounted returns computed for each trajectory
        k : int
            after step k action selection is unconfounded
        iter_ids : np.array, [num_iters]
                 indicator I = 1 if not prescribing antibiotics
        iter_ius : np.array, [num_iters]
                 indicator I((np.sqrt(self.max_horizon) * (u - 0.5)) 
            > self.confounding_threshold)      
        config : dictionary containing
            nS : int
                number of states
            nA : int
                number of actions
            discount : float
                discount factor
            max_horizon : int
                maximum number of timesteps in each simulation
            bootstrap : bool
                if use bootstrap
            n_bootstrap : int
                number of bootstrap samples (if bootstrap is True)
        Methods
        -------
        compute()
            computes the IS estimate for n_bootstrap
        _compute_is()
            computes is for one set of trajectories and returns
        compute_pd()
            computes the PDIS estimate for n_bootstrap
        _compute_pdis()
            computes pdis for one set of trajectories and returns
        _learn_split_policies()
            learn two behaviour policies, before k and after k
            _learn_first_k_step_policy()
            _learn_after_k_step_policy()
        """
        self.trajectories = trajectories
        self.returns = returns
        self.config = config
        self.iter_ids = iter_ids
        self.iter_ius = iter_ius
        self.k = k

    def compute(self, evaluation_policy, use_tqdm=True):
        '''compute
        computes the is estimate for evaluation policy
        (n_bootstrap times). This function learns the behaviour
        policies seprately for each bootstrap samples
        Parameters
        ----------
        evaluation_policy : (e_t0_policy, e_policy)
             tuple of two policies each [n_actions, n_states]
            - e_t0_policy : evaluation policy at step 0
            - e_policy : evaluation policy after step 1
        use_tqdm : bool
            if use tqdm
        Returns
        -------
        is_estimate : np.array, float [n_bootstrap]
            IS estimates
        '''
        if self.config['bootstrap']:
            is_estimate = np.zeros(self.config['n_bootstrap'])
            for i in tqdm(range(self.config['n_bootstrap']), disable=not use_tqdm):
                # bootstrap indexes:
                idxs = np.random.choice(np.arange(self.trajectories.shape[0]),
                            size=self.trajectories.shape[0], replace=True)
                obs = self.trajectories[idxs, :]
                returns = self.returns[idxs]
                ids = self.iter_ids[idxs]
                ius = self.iter_ius[idxs]
                # learn the behaviour policy
                b_t0_policy, b_policy = self._learn_split_policies(obs, ids)
                is_estimate[i] = self._compute_is(trajectories=obs, 
                            returns=returns, 
                            behaviour_policy = (b_t0_policy, b_policy), 
                            evaluation_policy=evaluation_policy, ius = ius)
        else:
            b_t0_policy, b_policy = self._learn_split_policies(self.trajectories, self.iter_ids)
            is_estimate = self._compute_is(trajectories=self.trajectories, 
                        returns=self.returns, 
                        behaviour_policy = (b_t0_policy, b_policy), 
                        evaluation_policy=evaluation_policy, ius = self.iter_ius)
        return is_estimate

    def _compute_is(self, trajectories, returns, behaviour_policy, evaluation_policy, ius):
        """_compute_is: adopted from David's paper
        Weighted Importance Sampling for Off Policy Evaluation
        Parameters
        ----------
        trajectories : np.array, float [None, max_horizon, 5]
            see __init__
        returns : np.array, float [None]
            see __init__
        behaviour_policy : (b_t0_policy, b_policy)
            b_t0_policy : np.array, float [n_actions, n_states]
                behaviour policy at step 0 
            b_policy : np.array, float [n_actions, n_states]
                behaviour policy after step 1
        evaluation_policy : (e_t0_policy, e_policy)
            e_t0_policy : np.array, float [n_actions, n_states]
                evaluation policy at step 0
            e_policy : np.array, float [n_actions, n_states]
                evaluation policy after step 1
        ius : np.array, [None]
                 indicator I((np.sqrt(self.max_horizon) * (u - 0.5)) 
            > self.confounding_threshold)          
        Returns
        -------
        is_est : float
            importance sampling estiamte
        """
        b_t0_policy, b_policy = behaviour_policy
        e_t0_policy, e_policy = evaluation_policy

        assert returns.ndim == 1 # check 1D array

        obs_actions = trajectories[..., 1].astype(int)
        obs_states = trajectories[..., 2].astype(int)
        obs_ius = ius.astype(int)

        # Evluation policy importance weights:
        # after first step
        p_eval = e_policy[obs_actions[:, 1:], obs_states[:, 1:]]
        
        # first step
        p_first_step = e_t0_policy[obs_actions[:,0],
                                    obs_states[:,0]]
        if len(p_first_step.shape) == 1:
            p_first_step = np.expand_dims(p_first_step, axis=-1)

        p_eval = np.concatenate([p_first_step, p_eval], axis=-1)

        # Behaviour policy importance weights:
        p_first_step = np.expand_dims(b_t0_policy[obs_actions[:,0], 
                                                obs_states[:,0], obs_ius], -1)
        p_behaviour = b_policy[obs_actions[:, 1:], obs_states[:, 1:]]
        p_behaviour = np.concatenate([p_first_step, p_behaviour], axis=-1)

        # Deal with variable length sequences by setting ratio to 1
        terminated_idx = obs_actions == -1
        p_behaviour[terminated_idx] = 1
        p_eval[terminated_idx] = 1

        assert np.all(p_behaviour > 0), "Some actions had zero prob under p_obs, IS fails"
        
        cum_ir = (p_eval / p_behaviour).prod(axis=1)
        is_idx = (cum_ir > 0)

        if is_idx.sum() == 0:
            print("Found zero matching IS samples, continuing")
            return np.nan

        is_est = (cum_ir) * returns
        return is_est.mean()
    
    def compute_pd(self, evaluation_policy, use_tqdm=True):
        '''compute
        computes the pdis estimate for evaluation policy
        (n_bootstrap times). This function learns the behaviour
        policies seprately for each bootstrap samples
        Parameters
        ----------
        evaluation_policy : (e_t0_policy, e_policy)
             tuple of two policies each [n_actions, n_states]
            - e_t0_policy : evaluation policy at step 0
            - e_policy : evaluation policy after step 1
        use_tqdm : bool
            if use tqdm
        Returns
        -------
        pdis_estimate : np.array, float [n_bootstrap]
            PDIS estimates
        '''
        if self.config['bootstrap']:
            pdis_estimate = np.zeros(self.config['n_bootstrap'])
            for i in tqdm(range(self.config['n_bootstrap']), disable=not use_tqdm):
                # bootstrap indexes:
                idxs = np.random.choice(np.arange(self.trajectories.shape[0]),
                            size=self.trajectories.shape[0], replace=True)
                obs = self.trajectories[idxs, :]
                returns = self.returns[idxs]
                ids = self.iter_ids[idxs]
                ius = self.iter_ius[idxs]
                # learn the behaviour policy
                b_t0_policy, b_policy = self._learn_split_policies(obs, ids)
                pdis_estimate[i] = self._compute_pdis(trajectories=obs, 
                            returns=returns, 
                            behaviour_policy = (b_t0_policy, b_policy), 
                            evaluation_policy=evaluation_policy, ius = ius)
        else:
            b_t0_policy, b_policy = self._learn_split_policies(self.trajectories, self.iter_ids)
            pdis_estimate = self._compute_pdis(trajectories=self.trajectories, 
                        returns=self.returns, 
                        behaviour_policy = (b_t0_policy, b_policy), 
                        evaluation_policy=evaluation_policy, ius = self.iter_ius)
        return pdis_estimate

    def _compute_pdis(self, trajectories, returns, behaviour_policy, evaluation_policy, ius):
        """_compute_pdis:
        ----------
        trajectories : np.array, float [None, max_horizon, 5]
            see __init__
        returns : np.array, float [None, max_horizon]
            see __init__
        behaviour_policy : (b_t0_policy, b_policy)
            b_t0_policy : np.array, float [n_actions, n_states]
                behaviour policy at step 0 
            b_policy : np.array, float [n_actions, n_states]
                behaviour policy after step 1
        evaluation_policy : (e_t0_policy, e_policy)
            e_t0_policy : np.array, float [n_actions, n_states]
                evaluation policy at step 0
            e_policy : np.array, float [n_actions, n_states]
                evaluation policy after step 1
        ius : np.array, [None]
                 indicator I((np.sqrt(self.max_horizon) * (u - 0.5)) 
            > self.confounding_threshold)
        Returns
        -------
        pdis_est : float
            per-decision importance sampling estiamte
        """
        b_t0_policy, b_policy = behaviour_policy
        e_t0_policy, e_policy = evaluation_policy

        obs_actions = trajectories[..., 1].astype(int)
        obs_states = trajectories[..., 2].astype(int)
        obs_ius = ius.astype(int)

        # Evluation policy importance weights:
        # after first step
        p_eval = e_policy[obs_actions[:, 1:], obs_states[:, 1:]]
        
        # first step
        p_first_step = e_t0_policy[obs_actions[:,0],
                                    obs_states[:,0]]
        if len(p_first_step.shape) == 1:
            p_first_step = np.expand_dims(p_first_step, axis=-1)

        p_eval = np.concatenate([p_first_step, p_eval], axis=-1)

        # Behaviour policy importance weights:
        p_first_step = np.expand_dims(b_t0_policy[obs_actions[:,0], 
                                                obs_states[:,0], obs_ius], -1)
        p_behaviour = b_policy[obs_actions[:, 1:], obs_states[:, 1:]]
        p_behaviour = np.concatenate([p_first_step, p_behaviour], axis=-1)

        # Deal with variable length sequences by setting ratio to 1
        terminated_idx = obs_actions == -1
        p_behaviour[terminated_idx] = 1
        p_eval[terminated_idx] = 1

        assert np.all(p_behaviour > 0), "Some actions had zero prob under p_obs, PDIS fails"
        
        cum_ir = (p_eval / p_behaviour).cumprod(axis=1)
        pdis_idx = (cum_ir > 0)

        if pdis_idx.sum() == 0:
            print("Found zero matching PDIS samples, continuing")
            return np.nan

        pdis = (cum_ir) * returns
        
        pdis = pdis.sum(axis = -1)

        pdis_est = pdis.mean()
        return pdis_est

    def _learn_split_policies(self, obs, ids):
        """
        learns two different policies, one before first k step, 
            and one after first k step
        Paremeters
        ----------
        obs : np.array, float [None, max_horizon, 5] 
        ids : np.array, [None]
                 indicator I = 1 if not prescribing antibiotics
        Returns
        -------
        (before_k, after_k) : tuple of policies
            before_k : np.array, floart [n_actions, n_states]
                behaviour policy before step k
            after_k : np.array, floart [n_actions, n_states]
                behaviour policy after step k
        """
        before_k = self._learn_first_k_step_policy(obs, ids)
        after_k = self._learn_after_k_step_policy(obs) 
        return (before_k, after_k)  

    def _learn_first_k_step_policy(self, obs, ids):
        """leanr policy before after step k
        
        Paremeters
        ----------
        obs : np.array, float [None, max_horizon, 5]
        ids : np.array, [None]
                 indicator I = 1 if not prescribing antibiotics   
        Returns
        -------
        policy : np.array, float [n_actions, n_states]
            learned policy
        """
        policies = np.zeros((self.config['nA'], self.config['nS'], 2))
        policies_u = np.zeros((self.config['nA'], self.config['nS'], 2))
        obs_w = obs[ids == 0]
        obs_wo = obs[ids == 1]
        
        for sample in range(obs_w.shape[0]):
            for step in range(obs_w.shape[1]):
                s = int(obs_w[sample, step, 2])
                a = int(obs_w[sample, step, 1])
                if a==-1 or step > self.k:
                    break
                policies[a, s, 0] += 1
        nonzero = policies[..., 0].sum(axis=0) > 0
        policies[:, nonzero, 0] /= policies[:, nonzero, 0].sum(axis=0, keepdims=True)
        
        for sample in range(obs_wo.shape[0]):
            for step in range(obs_wo.shape[1]):
                s = int(obs_wo[sample, step, 2])
                a = int(obs_wo[sample, step, 1])
                if a==-1 or step > self.k:
                    break
                policies[a, s, 1] += 1
        nonzero = policies[..., 1].sum(axis=0) > 0
        policies[:, nonzero, 1] /= policies[:, nonzero, 1].sum(axis=0, keepdims=True)
        
        Gamma = self.config['Gamma']
        
        policies_u[..., 0] = policies[..., 0] / (1 + np.sqrt(Gamma)) 
        + policies[..., 1] * np.sqrt(Gamma) / (1 + np.sqrt(Gamma))
        policies_u[..., 1] = policies[..., 0] * np.sqrt(Gamma) / (1 + np.sqrt(Gamma)) 
        + policies[..., 1] / (1 + np.sqrt(Gamma))
        
        return policies_u

    def _learn_after_k_step_policy(self, obs):
        """leanr policy after after step k
        Paremeters
        ----------
        obs : np.array, float [None, max_horizon, 5]
        
        Returns
        -------
        policy : np.array, float [n_actions, n_states]
            learned policy
        """
        policy = np.zeros((self.config['nA'], self.config['nS']))
        for sample in range(obs.shape[0]):
            for step in range(self.k, obs.shape[1]):
                s = int(obs[sample, step, 2])
                a = int(obs[sample, step, 1])
                if a==-1:
                    break
                policy[a, s] += 1
        nonzero = policy.sum(axis=0) > 0
        policy[:, nonzero] /= policy[:, nonzero].sum(axis=0, keepdims=True)
        return policy
