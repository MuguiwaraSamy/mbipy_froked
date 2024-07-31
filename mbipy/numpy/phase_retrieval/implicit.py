"""_summary_
"""

__all__ = ("lcs", "lcs_df","lcs_ddf")

import numpy as np
from scipy import ndimage

from ...src.phase_retrieval.implicit import (
    create_lcs,
    create_lcs_df,
    create_lcs_df_matrices,
    create_lcs_matrices,
    create_lcs_vectors,
    create_lcs_ddf,
    create_lcs_ddf_matrices,
    create_lcs_ddf_vectors,
)
from ...src.phase_retrieval.implicit.utils import (
    create_implicit_tracking,
    create_laplace,
    create_lstsq_solver,
    create_normal,
    create_normal_stack,
    create_tikhonov_stack,
)

normal_stack = create_normal_stack(np)
tikhonov_stack = create_tikhonov_stack(np)

normal = create_normal(np)

solvers = {"normal": normal}

lstsq_solver = create_lstsq_solver(solvers, normal_stack, tikhonov_stack)

implicit_tracking = create_implicit_tracking(
    np, np.lib.stride_tricks.sliding_window_view, lstsq_solver
)

laplace = create_laplace(np)

lcs_matrices = create_lcs_matrices(np)
lcs_vectors = create_lcs_vectors()


lcs = create_lcs(lcs_matrices, lcs_vectors, lstsq_solver, implicit_tracking)


lcs_df_matrices = create_lcs_df_matrices(np, laplace)

lcs_df = create_lcs_df(lcs_df_matrices, lcs_vectors, lstsq_solver)

lcs_ddf_matrices = create_lcs_ddf_matrices(np)
lcs_ddf_vectors = create_lcs_ddf_vectors()

lcs_ddf = create_lcs_ddf(lcs_ddf_matrices, lcs_ddf_vectors, lstsq_solver)

