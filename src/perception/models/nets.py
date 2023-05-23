import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from perception.models.sampling_schemes import sampling_registry


class MARONetwork(nn.Module):

    def __init__(self, n_agents, input_dim, output_obs_dim, hidden_dim, obs_processor=None):
        super(MARONetwork, self).__init__()

        # The input of the network is of shape input_dim.
        # The output of the network is of shape 2 * n_agents * output_obs_dim, where
        # the first n_agents * output_obs_dim positions encode the mean of the output
        # distribution for each of the agents, and the following n_agents * output_obs_dim
        # positions encode the std of the output distribution.

        # Args.
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.output_obs_dim = output_obs_dim
        self.hidden_dim = hidden_dim
        self.obs_processor = obs_processor

        # Layers.
        self.linear_1 = nn.Linear(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear_obs = nn.Linear(hidden_dim, 2 * self.output_obs_dim * n_agents) # mus and sigmas.

    def encode(self, latents, hidden):
        # (single-step).
        # x shape: [bs, obs_dim*n_agents] or [bs, unique_obs_dim*n_agents + common_obs_dim]

        batchsize = latents.shape[0]
        in_latents = latents.unsqueeze(1)  # [bs, 1, obs_dim*n_agents] or [bs, 1, unique_obs_dim*n_agents + common_obs_dim]

        outs = F.relu(self.linear_1(in_latents))
        outs, hidden = self.lstm(outs, hidden)
        outs_obs = self.linear_obs(outs)

        stride = self.output_obs_dim * self.n_agents
        obs_mus = outs_obs[:, :, :stride]
        obs_mus = obs_mus.view(batchsize, self.n_agents, self.output_obs_dim) # [bs,n_agents,output_obs_dim]
        obs_sigmas = outs_obs[:, :, stride:2 * stride]
        obs_sigmas = obs_sigmas.view(batchsize, self.n_agents, self.output_obs_dim)
        obs_sigmas = th.exp(obs_sigmas) # [bs,n_agents,output_obs_dim]

        return obs_mus, obs_sigmas, hidden

    def training_step(self, data, mask, train_params):

        # data shape: [bs, n_timesteps, n_agents, obs_dim]

        loss_info = {}

        if self.obs_processor:
            unique_obs, common_obs = self.obs_processor.split_obs(data)
            # unique_obs: [bs,n_timesteps,num_agent,unique_obs_dim]
            # common_obs: [bs,n_timesteps,num_agent,common_obs_dim]
            unique_obs = unique_obs.reshape(unique_obs.shape[0],unique_obs.shape[1], -1) # [bs,n_timesteps,unique_obs_dim*n_agents]
            net_input = th.cat([unique_obs, common_obs[:,:,0,:]], dim=-1) # [bs,n_timesteps,unique_obs_dim*n_agents+common_obs_dim]
        else:
            net_input = data.view(data.shape[0], data.shape[1], -1) # [bs,n_timesteps,obs_dim*n_agents]

        # Forward Pass.
        pred_mus, pred_sigmas = self.forward(net_input) # [bs,n_timesteps-1,n_agents,obs_dim] or [bs,n_timesteps-1,n_agents,unique_obs_dim]

        # Model Loss.
        x_next = th.roll(data.detach(), -1,  dims=1) # Roll. [bs,n_timesteps,n_agents,obs_dim]

        if self.obs_processor:
            unique_obs, _ = self.obs_processor.split_obs(data) # [bs,n_timesteps,num_agent,unique_obs_dim]
            unique_obs_next, _ = self.obs_processor.split_obs(x_next) # [bs,n_timesteps,num_agent,unique_obs_dim]
            deltas = unique_obs_next - unique_obs
        else:
            deltas = x_next - data # Compute deltas.
        
        deltas = deltas[:, :-1, :, :].clone().detach() # Drop last element

        obs_loss = self.training_loss(x_next=deltas,
                                      pred_mus=pred_mus,
                                      pred_sigmas=pred_sigmas,
                                      mask=mask,
                                      reduce=True)

        # Compute total_loss.
        loss_info['predictor_obs_loss'] = th.mean(obs_loss).cpu().item()

        return obs_loss, loss_info

    def forward(self, x):
        # Only for training purposes (multi-steps).
        # x shape: [bs, n_timesteps, n_agents*obs_dim] or [bs, n_timesteps, n_agents*unique_obs_dim + common_obs_dim]

        # Encode latents with the predictive model
        batchsize, seq_len = x.shape[0],x.shape[1]

        outs = F.relu(self.linear_1(x.clone().detach()))
        outs, hidden = self.lstm(outs)
        outs_obs = self.linear_obs(outs)

        stride = self.output_obs_dim * self.n_agents

        pred_mus = outs_obs[:, :, :stride]
        pred_mus = pred_mus.view(batchsize, seq_len, self.n_agents, self.output_obs_dim)
        pred_sigmas = outs_obs[:, :, stride:2 * stride]
        pred_sigmas = pred_sigmas.view(batchsize, seq_len, self.n_agents, self.output_obs_dim)
        pred_sigmas = th.exp(pred_sigmas)

        # Drop last element of the prediction model output.
        pred_mus    = pred_mus[:, :-1, :, :] # [bs,n_timesteps-1,n_agents,output_obs_dim]
        pred_sigmas = pred_sigmas[:, :-1, :, :] # [bs,n_timesteps-1,n_agents,output_obs_dim]

        return pred_mus, pred_sigmas

    def training_loss(self, x_next, pred_mus, pred_sigmas, mask, reduce=True):
        """ Computes loss.

        Computes the minus log probability of the batch under the model described
        by pred_mus, pred_sigmas. Precisely, bs1, bs2, ... are the sizes of the batch
        dimensions (several batch dimension are useful when you have both a batch
        axis and a time step axis), and fs is the number of features.

        :args x_next: (bs1, bs2, *, fs) torch tensor - next timestep data.
        :args pred_mus: (bs1, bs2, *, fs) torch tensor.
        :args pred_sigmas: (bs1, bs2, *, fs) torch tensor.
        :args mask: torch tensor.
        :args reduce: if not reduce, the mean in the following formula is ommited.

        :returns:
        loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...,} sum_{f=0..fs}
                log(N(x_next[i1, i2, ..., f] | mus[i1, i2, ..., f], sigmas[i1, i2, ..., f]))

        NOTE: The loss is not reduced along the feature dimension
            (i.e. it should scale ~linearly with fs).
        """
        # x_next shape: [bs,n_timesteps,n_agents,n_features]

        # Observation loss.
        normal_dist = Normal(pred_mus, pred_sigmas)
        g_log_probs = normal_dist.log_prob(x_next) # [bs,n_timesteps,n_agents,obs_dim]
        g_log_probs = g_log_probs.sum(-1) # [bs,n_timesteps,n_agents]
        prob_mask = mask[:, :-1, :].expand_as(g_log_probs).detach()
        obs_loss =  - (prob_mask * g_log_probs).sum(-1).sum(-1) / prob_mask.sum(-1).sum(-1) # [bs]
        obs_loss = th.mean(obs_loss) # []

        return obs_loss


class MARONetworkTeacherForcing(MARONetwork):

    def __init__(self, n_agents, input_dim, output_obs_dim, hidden_dim, obs_processor=None, train_comm_p=None):
        super(MARONetworkTeacherForcing, self).__init__(n_agents, input_dim, output_obs_dim, hidden_dim, obs_processor)
        self.train_comm_p = train_comm_p

    def training_step(self, data, mask, train_params):

        # data shape: [bs, n_timesteps, n_agents, obs_dim]
        bs, n_timesteps, n_agents, _ = data.shape

        if isinstance(self.train_comm_p, float):
            comm_p = self.train_comm_p
        elif isinstance(self.train_comm_p, str):
            comm_p = sampling_registry[self.train_comm_p]()
        else:
            raise ValueError("Incorrect sampling scheme selected:" + str(self.train_comm_p))

        loss_info = {}

        hidden_states = (th.zeros((1, bs, self.hidden_dim)), th.zeros((1, bs, self.hidden_dim)))

        pred_mus, pred_sigmas = [], []
        # First timestep (always communicate).
        if self.obs_processor:
            unique_obs, common_obs = self.obs_processor.split_obs(data[:,0,:,:])
            # unique_obs: [bs,num_agent,unique_obs_dim]
            # common_obs: [bs,num_agent,common_obs_dim]
            unique_obs = unique_obs.reshape(bs, -1) # [bs,unique_obs_dim*n_agents]
            net_input = th.cat([unique_obs, common_obs[:,0,:]], dim=-1) # [bs,unique_obs_dim*n_agents+common_obs_dim]
        else:
            net_input = data[:,0,:,:].view(bs, -1) # [bs,obs_dim*n_agents]
        
        pred_delta_mus, pred_delta_sigmas, hidden_states = self.encode(net_input, hidden=hidden_states) # [bs,n_agents,output_obs_dim]
        pred_mus.append(pred_delta_mus)
        pred_sigmas.append(pred_delta_sigmas)

        if self.obs_processor:
            unique_observations, _ = self.obs_processor.split_obs(data[:,0,:,:]) # [bs, n_agents, unique_obs_dim]
            last_pred_obs = unique_observations + pred_delta_mus # [bs, n_agents, unique_obs_dim]
        else:
            last_pred_obs = data[:,0,:,:] + pred_delta_mus # [bs, n_agents, obs_dim]
        last_pred_obs = last_pred_obs.detach()

        # For the other timesteps.
        for t in range(1, n_timesteps-1):

            # Generate mask given `comm_p` probability.
            comm_mask = th.rand(bs,n_agents) # [bs,num_agents]
            comm_mask = (comm_mask >= (1.0 - comm_p)) # Mask out entries.
            comm_mask = comm_mask.unsqueeze(2) # [bs,n_agents,1]

            if self.obs_processor:
                unique_obs, common_obs = self.obs_processor.split_obs(data[:,t,:,:])
                # unique_obs: [bs,num_agent,unique_obs_dim]
                # common_obs: [bs,num_agent,common_obs_dim]
                comm_mask = th.repeat_interleave(comm_mask, unique_obs.shape[-1], axis=-1) # [bs,n_agents,unique_obs_dim]
                mixed_unique_obs = th.where(comm_mask, unique_obs, last_pred_obs) # [bs,n_agents,unique_obs_dim]
                mixed_unique_obs = mixed_unique_obs.reshape(mixed_unique_obs.shape[0], -1) # [bs,n_agents*unique_obs_dim]
                net_input = th.cat([mixed_unique_obs, common_obs[:,0,:]], dim=-1) # [bs,unique_obs_dim*n_agents+common_obs_dim]
            else:
                comm_mask = th.repeat_interleave(comm_mask, data.shape[-1], axis=-1) # [bs,num_agents,obs_dim]
                mixed_obs = th.where(comm_mask, data[:,t,:,:], last_pred_obs) # [bs,num_agents,obs_dim]
                net_input = mixed_obs.view(bs, -1) # [bs,obs_dim*n_agents]

            pred_delta_mus, pred_delta_sigmas, hidden_states = self.encode(net_input, hidden=hidden_states) # [bs,n_agents,output_obs_dim]

            if self.obs_processor:
                unique_observations, _ = self.obs_processor.split_obs(data[:,t,:,:]) # [bs, n_agents, unique_obs_dim]
                last_pred_obs = unique_observations + pred_delta_mus # [bs, n_agents, unique_obs_dim]
            else:
                last_pred_obs = data[:,t,:,:] + pred_delta_mus # [bs, n_agents, obs_dim]
            last_pred_obs = last_pred_obs.detach()

            pred_mus.append(pred_delta_mus) 
            pred_sigmas.append(pred_delta_sigmas)

        pred_mus = th.stack(pred_mus, dim=1) # [bs, n_timesteps, n_agents, output_obs_dim]
        pred_sigmas = th.stack(pred_sigmas, dim=1) # [bs, n_timesteps, n_agents, output_obs_dim]

        # Model Loss.
        x_next = th.roll(data.detach(), -1,  dims=1) # Roll. [bs,n_timesteps,n_agents,obs_dim]

        if self.obs_processor:
            unique_obs, _ = self.obs_processor.split_obs(data) # [bs,n_timesteps,num_agent,unique_obs_dim]
            unique_obs_next, _ = self.obs_processor.split_obs(x_next) # [bs,n_timesteps,num_agent,unique_obs_dim]
            deltas = unique_obs_next - unique_obs
        else:
            deltas = x_next - data # Compute deltas.
        
        deltas = deltas[:, :-1, :, :].clone().detach() # Drop last element

        obs_loss = self.training_loss(x_next=deltas,
                                      pred_mus=pred_mus,
                                      pred_sigmas=pred_sigmas,
                                      mask=mask,
                                      reduce=True)

        # Compute total_loss.
        loss_info['predictor_obs_loss'] = th.mean(obs_loss).cpu().item()

        return obs_loss, loss_info

    def training_loss(self, x_next, pred_mus, pred_sigmas, mask, reduce=True):
        """ Computes loss.

        Computes the minus log probability of the batch under the model described
        by pred_mus, pred_sigmas. Precisely, bs1, bs2, ... are the sizes of the batch
        dimensions (several batch dimension are useful when you have both a batch
        axis and a time step axis), and fs is the number of features.

        :args x_next: (bs1, bs2, *, fs) torch tensor - next timestep data.
        :args pred_mus: (bs1, bs2, *, fs) torch tensor.
        :args pred_sigmas: (bs1, bs2, *, fs) torch tensor.
        :args mask: torch tensor.
        :args reduce: if not reduce, the mean in the following formula is ommited.

        :returns:
        loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...,} sum_{f=0..fs}
                log(N(x_next[i1, i2, ..., f] | mus[i1, i2, ..., f], sigmas[i1, i2, ..., f]))

        NOTE: The loss is not reduced along the feature dimension
            (i.e. it should scale ~linearly with fs).
        """
        # x_next shape: [bs,n_timesteps,n_agents,n_features]

        # Observation loss.
        normal_dist = Normal(pred_mus, pred_sigmas)
        g_log_probs = normal_dist.log_prob(x_next) # [bs,n_timesteps,n_agents,obs_dim]
        g_log_probs = g_log_probs.sum(-1) # [bs,n_timesteps,n_agents]
        prob_mask = mask[:, :-1, :].expand_as(g_log_probs).detach()
        obs_loss =  - (prob_mask * g_log_probs).sum(-1).sum(-1) / prob_mask.sum(-1).sum(-1) # [bs]
        obs_loss = th.mean(obs_loss) # []

        return obs_loss

