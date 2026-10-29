import sys
import numpy as np
from analyze import load, planned_at, wrap, fit_start
for f in sys.argv[1:]:
    d = load(f)
    lag = fit_start(d)
    tau = d["fb_t"] - d["times"][1] - lag
    err = np.degrees(np.abs(wrap(planned_at(d, tau) - d["fb_act"])))
    print(f.split("/")[-1], "fitted start lag ms", round(lag * 1000), "max err deg at that lag", err.max().round(3),
          "accept->first sample s", round(d["fb_t"][0] - d["times"][1], 3))
