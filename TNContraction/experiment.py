import GP.helper_functions as HF
from detect_commuties import optimize_pen
from circuits import build_toffoli_circuit_pos
import networkx as nx

if __name__ == '__main__':
    for pos in ['start','middle']:
        circ = build_toffoli_circuit_pos(pos=pos)

        #####################
        # Get graph from circuit
        G_toff = HF.tn_to_graph(circ.psi)
        # circ_start.psi.draw(color=['PSI0', 'CNOT', 'H','RZ','T','T_dagger'],show_tags=False)
        for k in [2,3,4,5,6,7,8]:
            # k = 6
            solution , pen = optimize_pen(G_toff,k,n_iter=10,hybrid=True)
            path = 'TNContraction/results/toffoli_'+pos+'_k'+str(k)+'_pen'+str(pen)
            HF.dict_list_to_json(path,[[solution]])