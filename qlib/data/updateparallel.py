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
        super(NewParallel, self).__init__(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            timeout=timeout,
            pre_dispatch=pre_dispatch,
            batch_size=batch_size,
            temp_folder=temp_folder,
            max_nbytes=max_nbytes,
            mmap_mode=mmap_mode,
            prefer=prefer,
            require=require,
        )
        if isinstance(self._backend, MultiprocessingBackend):
            self._backend_args["maxtasksperchild"] = maxtasksperchild






