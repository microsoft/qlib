from joblib import Parallel


class UpdateParallel(Parallel):
    def __init__(
        self,
        n_jobs=None,
        backend=None,
        verbose=0,
        timeout=None,
        pre_dispatch="2 * n_jobs",
        batch_size="auto",
        temp_folder=None,
        max_nbytes="1M",
        mmap_mode="r",
        prefer=None,
        require=None,
        maxtasksperchild=None,
        **kwargs
    ):
        super(UpdateParallel, self).__init__(n_jobs=n_jobs,
                                             backend=None,
                                             verbose=0,
                                             timeout=None,
                                             pre_dispatch="2 * n_jobs",
                                             batch_size="auto",
                                             temp_folder=None,
                                             max_nbytes="1M",
                                             mmap_mode="r",
                                             prefer=None,
                                             require=None,
                                             maxtasksperchild=None,
                                             **kwargs)
        self._backend_args["maxtasksperchild"] = ["maxtasksperchild"]

