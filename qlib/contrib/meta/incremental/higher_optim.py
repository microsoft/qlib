from higher import optim, patch
from higher.optim import _OverrideType, _GradCallbackType
import typing as _typing

import torch as _torch

import sys

if "_forward_pre_hooks" in patch._internal_attrs:
    patch._internal_attrs.remove("_forward_pre_hooks")


class DifferentiableOptimizer(optim.DifferentiableOptimizer):
    def step(
        self,
        input: _torch.Tensor,
        params: _typing.Iterable[_torch.Tensor] = None,
        override: _typing.Optional[_OverrideType] = None,
        grad_callback: _typing.Optional[_GradCallbackType] = None,
        first_order=False,
        **kwargs
    ) -> _typing.Iterable[_torch.Tensor]:
        r"""Perform a model update.

        This would be used by replacing the normal sequence::

            opt.zero_grad()
            loss.backward()
            opt.step()

        with::

            diffopt.step(loss)


        Args:
            loss: the loss tensor.
            params (optional): the parameters with regard to which we measure
                the loss. These must be provided if the differentiable optimizer
                did not receive a patched model with a view over its own fast
                weights at initialisation. If there is such a model, and params
                are provided, they will overwrite the params of the encapsulated
                model.
            override (optional): a dictionary mapping optimizer settings (i.e.
                those which would be passed to the optimizer constructor or
                provided within parameter groups) to either singleton lists of
                override values, or to a list of override values of length equal
                to the number of parameter groups. If a single override is
                provided for a keyword, it is used for all parameter groups. If
                a list is provided, the ``i``\ th element of the list overrides
                the corresponding setting in the ``i``\ th parameter group. This
                permits the passing of tensors requiring gradient to
                differentiable optimizers for use as optimizer settings. Setting
                override here has highest precedence, i.e. it will override any
                tensors provided as override during the creation of the
                differentiable optimizer, where there is name clash.
            grad_callback: (optional) a single argument function which will be
                applied to a list of gradients of parameters, which respects the
                order specified by ``reference_params``. This can be used to
                apply a function, such as gradient clipping, to all (or a
                subset) of these gradients every time the step function is
                called. This callback overrides the default provided when
                constructing the differentiable optimizer.


        Returns:
            The updated parameters, which will individually have ``grad_fn``\ s
            of their own. If the optimizer has an encapsulated patched model,
            its view over its own fast weights will be updated with these
            params.
        """

        # Deal with override
        if override is not None:
            self._apply_override(override)

        if self._fmodel is None or self._fmodel.fast_params is None:
            if params is None:
                raise ValueError(
                    "params kwarg must be passed to step if the differentiable "
                    "optimizer doesn't have a view on a patched model with "
                    "params."
                )
        else:
            params = self._fmodel.fast_params if params is None else params

        params = list(params)

        # if isinstance(input, _torch.Tensor):
        # This allows us to gracefully deal with cases where params are frozen.
        grad_targets = [p if p.requires_grad else _torch.tensor([], requires_grad=True) for p in params]
        all_grads = _torch.autograd.grad(
            input, grad_targets, create_graph=self._track_higher_grads, allow_unused=True,  # boo
        )
        if grad_callback is not None:
            all_grads = grad_callback(all_grads)
        elif self._grad_callback is not None:
            all_grads = self._grad_callback(all_grads)

        grouped_grads = []
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            grads = []
            for i, index in enumerate(mapping):
                group["params"][i] = params[index]
                grads.append(all_grads[index])
            grouped_grads.append(grads)
        # else:
        #     grouped_grads = []
        #     for _group, group, mapping in zip(input, self.param_groups, self._group_to_param_list):
        #         grads = []
        #         for i, index in enumerate(mapping):
        #             group['params'][i] = params[index]
        #             grads.append(input[index])
        #         grouped_grads.append(grads)

        self._update(grouped_grads)

        new_params = params[:]
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            for p, index in zip(group["params"], mapping):
                if not first_order:
                    # if self._track_higher_grads:
                    new_params[index] = p
                else:
                    new_params[index] = p.detach().requires_grad_()

        if self._fmodel is not None:
            self._fmodel.update_params(new_params)

        return new_params


setattr(
    sys.modules["higher.optim"].__dict__["DifferentiableOptimizer"], "step", DifferentiableOptimizer.step,
)
