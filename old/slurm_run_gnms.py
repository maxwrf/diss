import sys
from utils.config import params_from_json

# setup
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['gnm_build'])  # noqa

from GNMC import get_gnms

import numpy as np
from utils.config import params_from_json
from utils.gnm_utils import gnm_loopkup, gnm_rules


def main(in_path):
    """
    Computes the generative models for a single sample
    """
    data = np.load(in_path)
    A_init = data['A_init']
    A_Y = data['A_Y']
    D = data['D']
    params = data['params']
    div = data['div']
    region = data['region']

    # model x params x N ks statistics
    K_all = np.zeros((len(gnm_rules), params.shape[0], 4))

    # model x params
    K_max_all = np.zeros((len(gnm_rules), params.shape[0]))

    m = np.count_nonzero(A_Y) // 2

    # Run the generative models for this sample
    for j_model in range(len(gnm_rules)):
        model_num = gnm_loopkup[gnm_rules[j_model]]
        b, K = get_gnms(
            A_Y,
            A_init,
            D,
            params,
            int(m),
            int(model_num))

        K_all[j_model, ...] = K.T
        K_max_all[j_model, ...] = np.max(K, axis=0)

    # store
    out_path = in_path.replace('.dat', '.res')
    np.savez(out_path,
             div=div,
             region=region,
             K_all=K_all,
             K_max_all=K_max_all)

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)
