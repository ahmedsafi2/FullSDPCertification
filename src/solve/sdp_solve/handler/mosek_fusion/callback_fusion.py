import sys
import numpy
from mosek.fusion import *
from mosek import callbackcode, iinfitem, dinfitem, liinfitem

def makeUserCallback(model, maxtime):
    def userCallback(caller,
                     douinf,
                     intinf,
                     lintinf):
        opttime = 0.0

        if caller == callbackcode.begin_intpnt:
            pass
        elif caller == callbackcode.intpnt:
            itrn = intinf[iinfitem.intpnt_iter]
            pobj = douinf[dinfitem.intpnt_primal_obj]
            dobj = douinf[dinfitem.intpnt_dual_obj]
            stime = douinf[dinfitem.intpnt_time]
            opttime = douinf[dinfitem.optimizer_time]

        elif caller == callbackcode.end_intpnt:
            pass
        elif caller == callbackcode.begin_primal_simplex:
            pass
        elif caller == callbackcode.update_primal_simplex:
            itrn = intinf[iinfitem.sim_primal_iter]
            pobj = douinf[dinfitem.sim_obj]
            stime = douinf[dinfitem.sim_time]
            opttime = douinf[dinfitem.optimizer_time]

        elif caller == callbackcode.end_primal_simplex:
            pass
        elif caller == callbackcode.begin_dual_simplex:
            pass
        elif caller == callbackcode.update_dual_simplex:
            itrn = intinf[iinfitem.sim_dual_iter]
            pobj = douinf[dinfitem.sim_obj]
            stime = douinf[dinfitem.sim_time]
            opttime = douinf[dinfitem.optimizer_time]
        elif caller == callbackcode.end_dual_simplex:
            pass
        elif caller == callbackcode.begin_bi:
            pass
        elif caller == callbackcode.end_bi:
            pass
        else:
            pass

        if opttime >= maxtime:
            return 1
        return 0

    return userCallback

def userProgresCallback(caller):
    # Handle the caller code here
    pass