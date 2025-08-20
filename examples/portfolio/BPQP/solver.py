import operator
import time
from functools import reduce

import numpy as np
import osqp
import torch
import torch.nn.functional as F
import qpth.qp import QPFunction
from scipy import sparse
from torch import nn
from torch.autograd import Function

torch.set_default_dtype(torch.double)
device = "cuda" if torch.cuda.is_available() else "cpu"


def osqp_interface(P, q, A, lb, ub):
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False, eps_abs=1e-5, eps_rel=1e-5, eps_prim_inf=1e-5, eps_dual_inf=1e-5)
    t0 = time.time()
    res = prob.solve()
    return res.x, res.y, time.time() - t0


def qp_osqp_backward(x_value, y_value, P, G, A, grad_output):
    nineq, ndim = G.shape
    neq = A.shape[0]
    lambs = y_value[:nineq]  # active set
    active_set = np.concatenate([np.argwhere(lambs > 1e-4), np.argwhere(x_value <= 1e-4)])
    bG = G[active_set, :].squeeze()
    bb = np.zeros(neq)
    bh = np.zeros(len(active_set))
    bq = -grad_output.detach().cpu().numpy()
    osnewA = np.vstack([bG, A])
    osnewA = sparse.csc_matrix(osnewA)
    l_new = np.hstack([bh, bb])
    u_new = np.hstack([bh, bb])
    x_grad, y_grad, time_spent_backward = osqp_interface(P, bq, osnewA, l_new, u_new)
    return x_grad, y_grad, time_spent_backward


def create_qp_instances(P, q, G, h, A, b):
    P, q, G, h, A, b = [x.detach().cpu().numpy() for x in [P, q, G, h, A, b]]
    n_ineq = G.shape[0]
    P = sparse.csc_matrix(P)
    osA = np.vstack([G, A])
    osA = sparse.csc_matrix(osA)
    lb = np.hstack([np.zeros(n_ineq), 1.0]) # lower weight 0.
    ub = np.hstack([np.ones(n_ineq), 1.0]) # upper weight 0.5
    return P, q, osA, lb, ub


def BPQP(args, sign=-1):
    class BPQPmethod(Function):
        @staticmethod
        def forward(ctx, P, q):
            n_dim = P.shape[0]
            n_ineq = n_dim
            G = torch.diag_embed(torch.ones(n_dim)).to(device)
            h = torch.zeros(n_ineq).to(device)
            A = torch.ones(n_dim).unsqueeze(0).to(device)
            b = torch.tensor([1]).to(device)

            _P, _q, _osA, _l, _u = create_qp_instances(P, sign * q, G, h, A, b)
            x_value, y_value, _ = osqp_interface(_P, _q, _osA, _l, _u)
            ctx.P = _P
            ctx.G = G.cpu().numpy()
            ctx.A = A.cpu().numpy()
            yy = torch.cat(
                [
                    torch.from_numpy(x_value).to(device).to(torch.float32),
                    torch.from_numpy(y_value).to(device).to(torch.float32),
                ],
                dim=0,
            )

            ctx.save_for_backward(yy)
            return yy[:n_dim]

        @staticmethod
        def backward(ctx, grad_output):
            P, G, A = ctx.P, ctx.G, ctx.A
            ndim = P.shape[0]
            nineq = G.shape[0]
            yy = ctx.saved_tensors[0]
            x_star = yy[:ndim]
            lambda_star = yy[ndim: (ndim + nineq)]
            x_grad, _, _ = qp_osqp_backward(
                x_star.detach().cpu().numpy(), lambda_star.detach().cpu().numpy(), P, G, A, grad_output
            )
            try:
                x_grad = torch.from_numpy(x_grad).to(torch.float32).to(device)
            except TypeError:
                print('No solution')
                x_grad = None
            grads = (None, x_grad)
            return grads

    return BPQPmethod.apply


class CVXPY:
    pass


class NNSolver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        layer_sizes = [args["max_stock"] + 1, self._args["hiddenSize"], self._args["hiddenSize"]]
        layers = reduce(
            operator.add,
            [
                [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=args["dropout"])]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )
        output_dim = 1
        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, variance, x):
        try:
            x_in = torch.cat([variance, x.unsqueeze(1)], dim=1)
        except IndexError:
            print('Stock number < 1')
            x_in = torch.ones(variance.shape)*(1/variance.shape[0])
        x_in_m = torch.zeros((self._args["max_stock"], self._args["max_stock"] + 1)).to(device)
        x_in_m[: x_in.shape[0], : x_in.shape[1]] = x_in
        out = self.net(x_in_m).squeeze()
        if self._args["grad_step"]:
            G = torch.diag_embed(torch.ones(self._args["max_stock"])).to(device)
            h = torch.zeros(self._args["max_stock"]).to(device)
            A = torch.ones(self._args["max_stock"]).unsqueeze(0).to(device)
            b = torch.tensor([1]).to(device)
            out = grad_steps_all(out, A, b, G, h, self._args)
        return F.softmax(out[: len(x)])


def grad_steps_all(Y, A, b, G, h, args):
    lr = args["lr"]
    eps_converge = args["corrEps"]
    max_steps = args["corrTestMaxSteps"]
    momentum = args["corrMomentum"]
    Y_new = Y
    i = 0
    old_Y_step = 0
    with torch.no_grad():
        while (
                i == 0 or torch.max(torch.abs(A.mv(Y) - b)) > eps_converge or torch.max(G.mv(Y) - h) > eps_converge
        ) and i < max_steps:
            # Y_step = complete_partial(Y_new,A,b)
            ineq_step = ineq_grad(Y_new, G, h)
            eq_step = eq_grad(Y_new, A, b)
            Y_step = (1 - args["softWeightEqFrac"]) * ineq_step + args["softWeightEqFrac"] * eq_step

            new_Y_step = lr * Y_step + momentum * old_Y_step
            Y_new = Y_new - new_Y_step

            old_Y_step = new_Y_step
            i += 1

    return Y_new


def complete_partial(Y, A, b):
    rank_y = torch.linalg.matrix_rank(A)
    Z = torch.linalg.inv(A[:, :rank_y]) @ (b - Y @ A[:, rank_y:].T).T
    X = torch.cat([Z.T, Y], dim=1)
    return X


def eq_grad(Y, A, b):
    return 2 * (A.mv(Y) - b) @ A


def ineq_grad(Y, G, h):
    ineq_dist = G.mv(Y) - h
    return 2 * G.mv(ineq_dist)


def PDIPM(args, sign=-1):
    class QPfunction_gpuFn(Function):
        @staticmethod
        def forward(ctx, Q, q):
            nz = Q.shape[0]
            nineq = nz
            neq = 1
            initial_x, initial_lambs, initial_nu = (
                (1 / nz) * torch.ones(nz).to(device),
                torch.ones(nz).to(device),
                torch.zeros(1).to(device),
            )
            _G = -torch.diag_embed(torch.ones(nz)).to(device)
            _A = torch.ones(1, nz).to(device)
            _h = torch.zeros(nz).to(device)
            _b = torch.tensor(1).to(device)

            x_star, lambs_star, nu_star, _, k_inverse = torch_qp_solver(
                Q, sign * q, _G, _h, _A, _b, initial_x, initial_lambs, initial_nu, sigma=0.5, max_ite=500
            )

            ctx.nz, ctx.nineq, ctx.neq = nz, nineq, neq

            ctx.save_for_backward(x_star, lambs_star, nu_star, k_inverse)

            return x_star

        @staticmethod
        def backward(ctx, grad_output):
            nz, nineq, neq = ctx.nz, ctx.nineq, ctx.neq
            x_star, lambs_star, nu_star, k_inverse = ctx.saved_tensors
            loss_grad = torch.cat([grad_output, torch.zeros(nineq).to(device), torch.zeros(neq).to(device)], dim=0)
            dys = -k_inverse.mv(loss_grad)
            dzs = dys[:nz]
            dQs = 0.5 * (bger(dzs, x_star) + bger(x_star, dzs))
            dqs = dzs
            # Sparse Q,q
            dQs[dQs < 1e-8] = 0
            grads = (-dQs, -dqs)
            return grads

    return QPfunction_gpuFn.apply


def solve_kkt_r(Q, q, G, h, A, b, x, lambs, nu, elips):
    nineq = int(lambs.shape[0])
    r_dual = q + torch.mv(Q.T, x) + torch.mv(G.T, lambs) + torch.mv(A.T, nu)
    r_cent = torch.diag_embed(lambs).mv(torch.mv(G, x) - h).to(device) + torch.ones(nineq).to(device) * elips
    r_prim = torch.mv(A, x) - b
    return torch.cat([r_dual, r_cent, r_prim], dim=0)


def solve_grad_kkt_m(Q, G, h, A, x, lambs):
    nineq = int(G.shape[0])
    neq = int(A.shape[0])
    L1 = torch.cat([Q, G.T, A.T], dim=1)
    L2 = torch.cat(
        [torch.diag_embed(lambs).mm(G), torch.diag_embed(G.mv(x) - h), torch.zeros(nineq, neq).to(device)], dim=1
    )
    L3 = torch.cat([A, torch.zeros(neq, nineq).to(device), torch.zeros(neq, neq).to(device)], dim=1)
    return torch.cat([L1, L2, L3], dim=0)


def torch_qp_solver(Q, q, G, h, A, b, x, lambs, nu, sigma, max_ite):
    nz = int(Q.shape[0])
    nineq = int(G.shape[0])
    ita = -lambs.dot((G.mv(x) - h)) / nineq
    ita_store = []
    ita_store.append(ita)
    for _ in range(max_ite):
        elips = sigma * ita
        kkt = solve_kkt_r(Q=Q, q=q, G=G, h=h, A=A, b=b, x=x, lambs=lambs, nu=nu, elips=elips)
        L = solve_grad_kkt_m(Q=Q, G=G, h=h, A=A, x=x, lambs=lambs)
        # 1e-5 lower set to 0
        k_inverse = torch.linalg.inv(L)
        delta_y = k_inverse.mv(-kkt)  # update delta_y
        delta_lambs = delta_y[nz: (nz + nineq)]
        try:
            s_max = min(1, min(-lambs[delta_lambs < 0] / delta_lambs[delta_lambs < 0]))
        except TypeError:
            s_max = 1
        s = 0.99 * s_max
        x_trail = x + s * delta_y[:nz]
        while max(G.mv(x_trail) - h) >= 0:
            s = 0.5 * s
            x_trail = x + s * delta_y[:nz]

        lambs_trail = lambs + s * delta_y[nz: (nz + nineq)]
        nu_trail = nu + s * delta_y[(nz + nineq):]
        kkt_trail = solve_kkt_r(Q=Q, q=q, G=G, h=h, A=A, b=b, x=x_trail, lambs=lambs_trail, nu=nu_trail, elips=elips)
        while torch.norm(kkt_trail) > (1 - 0.1 * s) * torch.norm(kkt):
            s = 0.5 * s
            x_trail = x + s * delta_y[:nz]
            lambs_trail = lambs + s * delta_y[nz: (nz + nineq)]
            nu_trail = nu + s * delta_y[(nz + nineq):]
            kkt_trail = solve_kkt_r(
                Q=Q, q=q, G=G, h=h, A=A, b=b, x=x_trail, lambs=lambs_trail, nu=nu_trail, elips=elips
            )  # Last KKT
        x = x + s * delta_y[:nz]
        lambs = lambs + s * delta_y[nz: (nz + nineq)]
        nu = nu + s * delta_y[(nz + nineq):]

        ita = -lambs.dot((G.mv(x) - h)) / nineq
        ita_store.append(ita)
        if (ita < 1e-3) and torch.sqrt(
                torch.norm(delta_y[:nz], p=2) ** 2 + torch.norm(delta_y[(nz + nineq):], p=2) ** 2
        ) < 1e-3:
            break
    return x, lambs, nu, ita_store, k_inverse


def bger(x, y):
    return x.unsqueeze(1).mm(y.unsqueeze(1).T)


def QPTH(P, q, sign=-1):
    # qpth tend to yield unsolved solution
    nineq, ndim = P.shape
    P = P + torch.eye(ndim, ndim).to(device) * 1e-6
    G = -torch.diag_embed(torch.ones(ndim)).to(device)
    A = torch.ones(1, ndim).to(device)
    h = torch.zeros(ndim).to(device)
    b = torch.tensor(1).to(device)
    qpf = QPFunction(verbose=0, maxIter=500)
    try:
        qpth_x_value = qpf(P, sign * q, G, h, A, b)
    except TypeError:
        return F.softmax(q)
    return qpth_x_value.squeeze()
