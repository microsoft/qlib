import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/us_data")
from qlib.data import D

instruments = ["a1x4w7"]
fields = ["PSum($$q_taxrate, 4)/$close"]
print(D.features(instruments, fields, start_time="2020-06-01", end_time="2020-06-10", freq="day"))
