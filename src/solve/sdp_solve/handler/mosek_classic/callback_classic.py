import mosek


callback_codes = [
    "begin_bi",
    "begin_conic",
    "begin_dual_bi",
    "begin_dual_sensitivity",
    "begin_dual_setup_bi",
    "begin_dual_simplex",
    "begin_dual_simplex_bi",
    "begin_folding",
    "begin_folding_bi",
    "begin_folding_bi_dual",
    "begin_folding_bi_initialize",
    "begin_folding_bi_optimizer",
    "begin_folding_bi_primal",
    "begin_infeas_ana",
    "begin_initialize_bi",
    "begin_intpnt",
    "begin_license_wait",
    "begin_mio",
    "begin_optimize_bi",
    "begin_optimizer",
    "begin_presolve",
    "begin_primal_bi",
    "begin_primal_repair",
    "begin_primal_sensitivity",
    "begin_primal_setup_bi",
    "begin_primal_simplex",
    "begin_primal_simplex_bi",
    "begin_qcqo_reformulate",
    "begin_read",
    "begin_root_cutgen",
    "begin_simplex",
    "begin_solve_root_relax",
    "begin_to_conic",
    "begin_write",
    "conic",
    "decomp_mio",
    "dual_simplex",
    "end_bi",
    "end_conic",
    "end_dual_bi",
    "end_dual_sensitivity",
    "end_dual_setup_bi",
    "end_dual_simplex",
    "end_dual_simplex_bi",
    "end_folding",
    "end_folding_bi",
    "end_folding_bi_dual",
    "end_folding_bi_initialize",
    "end_folding_bi_optimizer",
    "end_folding_bi_primal",
    "end_infeas_ana",
    "end_initialize_bi",
    "end_intpnt",
    "end_license_wait",
    "end_mio",
    "end_optimize_bi",
    "end_optimizer",
    "end_presolve",
    "end_primal_bi",
    "end_primal_repair",
    "end_primal_sensitivity",
    "end_primal_setup_bi",
    "end_primal_simplex",
    "end_primal_simplex_bi",
    "end_qcqo_reformulate",
    "end_read",
    "end_root_cutgen",
    "end_simplex",
    "end_simplex_bi",
    "end_solve_root_relax",
    "end_to_conic",
    "end_write",
    "folding_bi_dual",
    "folding_bi_optimizer",
    "folding_bi_primal",
    "heartbeat",
    "im_dual_sensivity",
    "im_dual_simplex",
    "im_license_wait",
    "im_lu",
    "im_mio",
    "im_mio_dual_simplex",
    "im_mio_intpnt",
    "im_mio_primal_simplex",
    "im_order",
    "im_primal_sensivity",
    "im_primal_simplex",
    "im_read",
    "im_root_cutgen",
    "im_simplex",
    "intpnt",
    "new_int_mio",
    "optimize_bi",
    "primal_simplex",
    "qo_reformulate",
    "read_opf",
    "read_opf_section",
    "restart_mio",
    "solving_remote",
    "update_dual_bi",
    "update_dual_simplex",
    "update_dual_simplex_bi",
    "update_presolve",
    "update_primal_bi",
    "update_primal_simplex",
    "update_primal_simplex_bi",
    "update_simplex",
    "write_opf",
]


def makeUserCallback(maxtime, task):
    pass

    def userCallback(caller, douinf, intinf, lintinf):
        opttime = 0.0

        code = callback_codes[caller]
        
        if caller == mosek.callbackcode.begin_optimizer:
            pass
        elif caller == mosek.callbackcode.end_optimizer:
            pass
        # INTERIOR-POINT
        elif caller == mosek.callbackcode.begin_intpnt:
            pass
        elif caller == mosek.callbackcode.intpnt:
            itrn = intinf[mosek.iinfitem.intpnt_iter]
            pobj = douinf[mosek.dinfitem.intpnt_primal_obj]
            dobj = douinf[mosek.dinfitem.intpnt_dual_obj]
            stime = douinf[mosek.dinfitem.intpnt_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]

        elif caller == mosek.callbackcode.end_intpnt:
            pass

        # SIMPLEX
        elif caller == mosek.callbackcode.begin_simplex:
            pass
        elif caller == mosek.callbackcode.begin_primal_simplex:
            pass
        elif caller == mosek.callbackcode.update_primal_simplex:
            itrn = intinf[mosek.iinfitem.sim_primal_iter]
            pobj = douinf[mosek.dinfitem.sim_obj]
            stime = douinf[mosek.dinfitem.sim_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]

        elif caller == mosek.callbackcode.end_primal_simplex:
            pass
        elif caller == mosek.callbackcode.begin_dual_simplex:
            pass
        elif caller == mosek.callbackcode.update_dual_simplex:
            itrn = intinf[mosek.iinfitem.sim_dual_iter]
            pobj = douinf[mosek.dinfitem.sim_obj]
            stime = douinf[mosek.dinfitem.sim_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]
        elif caller == mosek.callbackcode.end_dual_simplex:
            pass

        # CONIC
        elif caller == mosek.callbackcode.begin_conic:
            pass
        elif caller == mosek.callbackcode.conic:

            itrn = intinf[mosek.iinfitem.intpnt_iter]

            pobj = douinf[mosek.dinfitem.intpnt_primal_obj]
            dobj = douinf[mosek.dinfitem.intpnt_dual_obj]

            # stime = douinf[mosek.dinfitem.conic_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]

        elif caller == mosek.callbackcode.end_conic:
            pass

        # PRESOLVE
        elif caller == mosek.callbackcode.begin_presolve:
            pass
        elif caller == mosek.callbackcode.heartbeat:
            pass
        elif caller == mosek.callbackcode.end_presolve:
            pass

        # MIO
        elif caller == mosek.callbackcode.new_int_mio:
            xx = task.getxx(mosek.soltype.itg)
        else:
            pass

        if opttime >= maxtime:
            # mosek is spending too much time. Terminate it.
            return 1

        return 0

    return userCallback
