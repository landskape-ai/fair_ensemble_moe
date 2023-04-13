import copy
from distutils.command.config import config

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
from torch.nn import functional as F

from .utils import ACT2FN

# -------------------#
# MoE


class SMoE_MLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout, hidden_act, state_dict=None
    ):
        super(SMoE_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[hidden_act]
        self.log_soft = nn.LogSoftmax(1)  # (1)

        if state_dict is not None:
            state_dict = copy.deepcopy(state_dict)
            self.fc1.weight = nn.Parameter(state_dict["fc1.weight"], requires_grad=True)
            self.fc1.bias = nn.Parameter(state_dict["fc1.bias"], requires_grad=True)
            self.fc2.weight = nn.Parameter(state_dict["fc2.weight"], requires_grad=True)
            self.fc2.bias = nn.Parameter(state_dict["fc2.bias"], requires_grad=True)
            for p in self.parameters():
                p.requires_grad = True

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.log_soft(out)
        return out


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0))  # .numpy()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        device = stitched.device

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SMoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        num_experts,
        input_size,
        hidden_size,
        noisy_gating,
        k,
        dropout,
        hidden_act,
        loss_coef=1e-2,
        routing="topk",
        capacity_factor=1.0,
        nb_routers=1,
        state_dict=None,
        cosine_reg=False,
        gamma_cosine=0.3,
        zeta_cosine=0.4,
        nb_tasks=None,
        task_il=False,
        **kwargs
    ):
        super(SMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.loss_coef = loss_coef
        self.routing = routing
        self.capacity_factor = capacity_factor
        self.nb_routers = nb_routers
        self.cosine_reg = cosine_reg
        self.gamma_cosine = gamma_cosine
        self.zeta_cosine = zeta_cosine
        self.nb_tasks = nb_tasks
        self.routing_fn = (
            self.noisy_top_k_gating if routing == "topk" else self.hard_task_gating if routing == "hard-task" else self.switch_gating if routing == "switch" else self.expert_choice_gating
        )

        # instantiate experts
        self.experts = nn.ModuleList(
            [
                SMoE_MLP(
                    self.input_size,
                    self.hidden_size,
                    self.dropout,
                    self.hidden_act,
                    state_dict,
                )
                for _ in range(self.num_experts)
            ]
        )
        if self.nb_tasks is None or not task_il:
            self.w_gate = nn.Linear(self.input_size, self.num_experts, bias=False)
        elif routing == "hard-task":
            probs = torch.rand(self.nb_tasks, self.num_experts, requires_grad=False)
            expert_mask = (probs >= probs.max(dim=1, keepdim=True)[0]).float()
            self.w_gate = nn.Parameter(expert_mask, requires_grad=False)
        else:
            self.w_gate = nn.ModuleList(
                [
                    nn.Linear(self.input_size, self.num_experts, bias=False)
                    for _ in range(self.nb_tasks)
                ]
            )

        self.w_noise = nn.Linear(self.input_size, self.num_experts, bias=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.apply(self._init_weights)

        assert self.k <= self.num_experts

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.kaiming_uniform_(m)
        if isinstance(m, nn.ModuleList):
            for module in m:
                self._init_weights(module)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, task=None):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        # clean_logits = torch.einsum('...bnd,...de->...bne', x, self.w_gate)
        if task is not None:
            if not isinstance(task, int):
                clean_logits = torch.stack([self.w_gate[i](x[i]) for i in task])
            else:
                clean_logits = self.w_gate[task](x)
        else:
            clean_logits = self.w_gate(x)
        if self.noisy_gating:
            raw_noise_stddev = self.w_noise(
                x
            )  # torch.einsum('...bnd,...de->...bne', x, self.w_noise)
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon) * train
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts), dim=-1
        )  # dim=1
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        device = top_k_gates.device
        zeros = torch.zeros_like(logits, requires_grad=True, device=device)

        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        soft_zeros = torch.zeros_like(logits, requires_grad=True, device=device)
        logsoft_gates = soft_zeros.scatter(
            1, top_k_indices, torch.log_softmax(top_k_logits, dim=1)
        )

        if self.noisy_gating and self.k < self.num_experts:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, logsoft_gates

    def hard_task_gating(self, x, train, task=None):
        """ Hard task gating 
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          Returns:
          x_out: a Tensor with shape [batch_size, input_size]
          load: a Tensor with shape [num_experts]
          """
        assert task is not None
        if isinstance(task, int):
            task_expert = self.w_gate[task].argmax().int()
            X_out = self.experts[task_expert](x)
            load = self.w_gate[task]
        else:
            task_indices, batch_indices = torch.sort(task) # sort task indices
            unique_task, n_per_task = torch.unique(task_indices, sorted=True, return_counts=True)
            task_indices = torch.split(batch_indices, n_per_task.tolist())
            X_out = torch.cat([self.experts[self.w_gate[unique_task[t]].argmax().int()](x[task_indices[t]]) for t in range(len(unique_task))], dim=0)
            # return the original order
            _, reverse_indices = torch.sort(batch_indices)
            X_out = X_out[reverse_indices]
            load = (self.w_gate[unique_task] * n_per_task.unsqueeze(1)).mean(dim=0)
        return X_out, load

    def _switch_loss(self, logits, e_mask, alpha=1e-2):
        # logits: n e
        # e_mask: n e
        # loss: scalar
        density_1 = torch.mean(e_mask.float(), dim=0).squeeze().unsqueeze(1)
        density_1_proxy = torch.mean(logits, dim=0).squeeze().unsqueeze(0)
        loss = alpha * torch.einsum("ne, en -> n", density_1_proxy, density_1) * self.num_experts
        return loss, density_1.squeeze()

    def switch_gating(self, x, train, task=None, eps=1e-2):
        T = round(x.size(0) * self.capacity_factor / self.num_experts)

        if task is not None:
            if not isinstance(task, int):
                clean_logits = torch.stack([self.w_gate[i](x[i]) for i in task])
            else:
                clean_logits = self.w_gate[task](x)
    
        else:
            clean_logits = self.w_gate(x)  # torch.einsum('...nd,...de->...ne', x, self.w_gate)

        if train and self.noisy_gating:
            clean_logits += (-2*eps) * torch.rand_like(clean_logits) + 1 + eps

        logits = self.softmax(clean_logits) # n e
        e_gates, e_index = logits.topk(k=1, dim=-1) # n 1
        e_index = e_index.squeeze()
        e_mask = F.one_hot(e_index, self.num_experts) # n e

        pos_in_e = torch.cumsum(e_mask, dim=0) * e_mask

        e_mask *= torch.less(pos_in_e, T)

        aux_loss, load = self._switch_loss(logits, e_mask)

        dropped_tokens = torch.ge(pos_in_e, T).sum()

        e_mask_flat = torch.sum(e_mask, dim=1)

        e_gates = e_gates.unsqueeze(2)
        e_index_hot = F.one_hot(e_index, self.num_experts).unsqueeze(2)
        e_mask_flat = e_mask_flat.unsqueeze(1).unsqueeze(1)
        e_mask_hot = F.one_hot(e_mask, T)

        combine_tensor = e_gates * e_mask_flat * e_index_hot *  e_mask_hot # (n, e, capacity)
        combine_tensor = rearrange(combine_tensor, "n e c -> e c n")
        dispatch_tensor = combine_tensor.bool() # (e, capacity, n) bool

        X_in = torch.einsum("...n,...nd->...d", dispatch_tensor.float(), x) # (e,c,d)
        X_e = torch.stack(
            [self.experts[i](X_in[i]) for i in range(self.num_experts)], dim=0
        ) # (e, c, d)
        X_out = torch.einsum("ecn,ecd -> nd", combine_tensor, X_e)
        return X_out, e_gates, aux_loss, load, dropped_tokens

    def expert_choice_gating(self, x, train, task=None):
        """expert choice gating: select top T tokens"""
        T = round(x.size(0) * self.capacity_factor / self.num_experts)
        if task is not None:
            if not isinstance(task, int):
                clean_logits = torch.stack([self.w_gate[i](x[i]) for i in task])
            else:
                clean_logits = self.w_gate[task](x)
        else:
            clean_logits = self.w_gate(x)  # torch.einsum('...nd,...de->...ne', x, self.w_gate)
        gates = self.softmax(clean_logits)
        logsoft_gates = torch.log_softmax(clean_logits, dim=-1)
        S = gates.T  # n x e
        top_t_gates, top_t_indices = S.topk(T, dim=-1)  # e x T
        P = F.one_hot(top_t_indices, x.size(0)).float()  # e x T x n
        X_in = torch.einsum("...n,...nd->...d", P, x)  # e x T x d
        X_e = torch.stack(
            [self.experts[i](X_in[i]) for i in range(self.num_experts)], dim=0
        )  # e x T x d
        # X_out will combine P, G, X_e to get the final output with shape n x d
        # print(P.shape, top_t_gates.shape, X_e.shape)
        X_out = torch.einsum("ekn,ek,ekd->nd", P, top_t_gates, X_e)
        # print(X_out.shape, "X_out")
        return X_out, gates, logsoft_gates, X_e

    def _calc_cosine_loss(self, X_e):
        X_e_t = rearrange(X_e, "e t d -> e d t")
        X_e_pairs_cosine_sim = torch.einsum("ejd,edl->ejl", X_e, X_e_t) / (torch.norm(X_e, dim=-1) ** 2 + 1e-8).unsqueeze(-1)
        eye = torch.eye(X_e_pairs_cosine_sim.size(1)).to(X_e_pairs_cosine_sim.device).unsqueeze(0)
        X_e_pairs_cosine_sim = X_e_pairs_cosine_sim - eye
        # loss = (X_e_pairs_cosine_sim**2).mean() # option 1
        loss = self.zeta_cosine * F.relu(X_e_pairs_cosine_sim - self.gamma_cosine).mean() # option 2
        return loss

    def forward(self, x, train=True, return_gates=False, task=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        b = x.size(0)
        x = rearrange(x, "b n d -> (b n) d")
        if not isinstance(task, int) and task is not None:
            task = task.repeat_interleave(x.size(0) // task.size(0))

        if self.routing == "hard-task":
            x_out, load = self.routing_fn(x, train, task=task)
            x_out = rearrange(x_out, "(b n) d -> b n d", b=b)
            return x_out, 0, 0, load, 0, 0

        if self.routing == "switch":
            x_out, gates, aux_loss, load, dropped_tokens = self.routing_fn(x, train, task=task)
            x_out = rearrange(x_out, "(b n) d -> b n d", b=b)
            return x_out, 0, gates, load, 0, aux_loss, dropped_tokens

        if self.routing == "expert-choice":
            x_out, gates, logsoft_gates, X_e = self.routing_fn(x, train, task=task)
            if self.cosine_reg:
                cosine_loss = self._calc_cosine_loss(X_e)
            else:
                cosine_loss = 0
            x_out = rearrange(x_out, "(b n) d -> b n d", b=b)
            importance = gates.sum(0)
            return x_out, 0, importance, 0, logsoft_gates, cosine_loss

        gates, load, logsoft_gates = self.routing_fn(x, train, task=task)

        # calculate importance loss
        importance = gates.sum(0)

        # calculate loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # print([expert_inputs[i].shape for i in range(self.num_experts)])
        gates_ = dispatcher.expert_to_gates()

        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        y = rearrange(y, "(b n) d -> b n d", b=b)
        if return_gates:
            return y, loss, importance, load, logsoft_gates, 0
        else:
            return y, loss, importance, load, None, 0
