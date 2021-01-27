from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence
from tianshou.data import Batch
import numpy as np
import torch
import copy
from typing import Union, Optional
from numbers import Number


def nan_weighted_avg(vals, weights, axis=None):
    """

    :param vals: The values to be averaged on.
    :param weights: The weights of weighted avrage.
    :param axis: On which axis to calculate the weighted avrage. (Default value = None)

    """
    assert vals.shape == weights.shape, AssertionError(f"{vals.shape} & {weights.shape}")
    vals = vals.copy()
    weights = weights.copy()
    res = (vals * weights).sum(axis=axis) / weights.sum(axis=axis)
    return np.nan_to_num(res, nan=vals[0])


def robust_auc(y_true, y_pred):
    """

    Calculate AUC.

    """
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return np.nan


def merge_dicts(d1, d2):
    """

    :param d1: Dict 1.
    :type d1: dict
    :param d2: Dict 2.
    :returns: A new dict that is d1 and d2 deep merged.
    :rtype: dict

    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(
    original, new_dict, new_keys_allowed=False, whitelist=None, override_all_if_type_changes=None,
):
    """Updates original dict with values from new_dict recursively.
    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the whitelist, then new subkeys can be introduced.

    :param original: Dictionary with default values.
    :type original: dict
    :param new_dict(dict: dict): Dictionary with values to be updated
    :param new_keys_allowed: Whether new keys are allowed. (Default value = False)
    :type new_keys_allowed: bool
    :param whitelist: List of keys that correspond to dict
    values where new subkeys can be introduced. This is only at the top
    level. (Default value = None)
    :type whitelist: Optional[List[str]]
    :param override_all_if_type_changes: List of top level
    keys with value=dict, for which we always simply override the
    entire value (dict), iff the "type" key in that value dict changes. (Default value = None)
    :type override_all_if_type_changes: Optional[List[str]]
    :param new_dict:

    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if (
                k in override_all_if_type_changes
                and "type" in value
                and "type" in original[k]
                and value["type"] != original[k]["type"]
            ):
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def get_seqlen(done_seq):
    """

    :param done_seq:

    """
    seqlen = []
    length = 0
    for i, done in enumerate(done_seq):
        length += 1
        if done:
            seqlen.append(length)
            length = 0
    if length > 0:
        seqlen.append(length)
    return np.array(seqlen)


def generate_seq(seqlen, list):
    """

    :param seqlen: param list:
    :param list:

    """
    res = []
    index = 0
    maxlen = np.max(seqlen)
    for i in seqlen:
        if isinstance(list, torch.Tensor):
            res.append(torch.cat((list[index : index + i], torch.zeros_like(list[: maxlen - i])), dim=0,))
        else:
            res.append(np.concatenate((list[index : index + i], np.zeros_like(list[: maxlen - i])), axis=0))
        index += i
    if isinstance(list, torch.Tensor):
        res = torch.stack(res, dim=0)
    else:
        res = np.stack(res, axis=0)
    return res


def sequence_batch(batch):
    """

    :param batch:

    """
    seqlen = get_seqlen(batch.done)
    # print(seqlen.max())
    # print(len(seqlen))
    res = Batch()
    # print(batch.keys())

    for v in batch.keys():
        if v not in ["policy", "info"]:
            res[v] = generate_seq(seqlen, batch[v])
        else:
            res[v] = batch[v]
    res.seqlen = seqlen
    return res


def flatten_seq(seq, seqlen):
    """

    :param seq: param seqlen:
    :param seqlen:

    """
    res = []
    for i, length in enumerate(seqlen):
        res.append(seq[i][:length])
    if isinstance(seq, torch.Tensor):
        res = torch.cat(res, dim=0)
    else:
        res = np.concatenate(res, axis=0)

    return res


def flatten_batch(batch):
    """

    :param batch:

    """
    for v in batch.keys():
        if v in ["policy", "info", "seqlen"]:
            continue
        batch[v] = flatten_seq(batch[v], batch.seqlen)
    return batch


def to_numpy(
    x: Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor]
) -> Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor]:
    """

    :param x: Union[Batch:
    :param dict: param list:
    :param tuple: param np.ndarray:
    :param torch: Tensor]:
    :param x: Union[Batch:
    :param list:
    :param np.ndarray:
    :param torch.Tensor]:
    :param x: Union[Batch:

    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    elif isinstance(x, Batch):
        x.to_numpy()
    elif isinstance(x, (list, tuple)):
        try:
            x = to_numpy(_parse_value(x))
        except TypeError:
            x = [to_numpy(e) for e in x]
    else:  # fallback
        x = np.asanyarray(x)
    return x


def to_torch(
    x: Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor]:
    """

    :param x: Union[Batch:
    :param dict: param list:
    :param tuple: param np.ndarray:
    :param torch: Tensor]:
    :param dtype: Optional[torch.dtype]:  (Default value = None)
    :param device: Union[str:
    :param int: param torch.device]:  (Default value = 'cpu')
    :param x: Union[Batch:
    :param list:
    :param np.ndarray:
    :param torch.Tensor]:
    :param dtype: Optional[torch.dtype]:  (Default value = None)
    :param device: Union[str:
    :param torch.device]: (Default value = 'cpu')
    :param x: Union[Batch:
    :param dtype: Optional[torch.dtype]:  (Default value = None)
    :param device: Union[str:

    """
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_torch(v, dtype, device)
    elif isinstance(x, Batch):
        x.to_torch(dtype, device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        x = to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (list, tuple)):
        try:
            x = to_torch(_parse_value(x), dtype, device)
        except TypeError:
            x = [to_torch(e, dtype, device) for e in x]
    else:  # fallback
        x = np.asanyarray(x)
        if issubclass(x.dtype.type, (np.bool_, np.number)):
            x = torch.from_numpy(x).to(device)
            if dtype is not None:
                x = x.type(dtype)
        else:
            raise TypeError(f"object {x} cannot be converted to torch.")
    return x


def to_torch_as(x: Union[torch.Tensor, dict, Batch, np.ndarray], y: torch.Tensor) -> Union[dict, Batch, torch.Tensor]:
    """

    :param x: Union[torch.Tensor:
    :param dict: param Batch:
    :param np: ndarray]:
    :param y: torch.Tensor:
    :param x: Union[torch.Tensor:
    :param Batch:
    :param np.ndarray]:
    :param y: torch.Tensor:
    :param x: Union[torch.Tensor:
    :param y: torch.Tensor:
    :returns: to_torch(x, dtype=y.dtype, device=y.device)``.

    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)
