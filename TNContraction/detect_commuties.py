import GP.graph_partitioning as GP
import GP.helper_functions as HF


def optimize_pen(G,k,n_iter=10,n_fix=5,hybrid=False,pen=0,narrow=False,sparse=False,check='random'):
    GraphP = GP.GraphPartition(G,k,sparse=sparse)
    n = G.number_of_nodes()
        
    scaling = [0.99,0.9,1.5,1.25]
    if narrow:
        scaling = [0.99,0.98,1.1,1.03]

    # Initial guess for the penalty if not
    #already provided 
    if not pen: # pen=0 is false, otherwise true
        pen = 1/n
    best_pen = pen
    
    # Initial step
    if hybrid:
        sampleset = GraphP.hybrid_solve(pen)
    else:
        sampleset = GraphP.quantum_solve(pen)

    # Fetch results from initial step
    sols = HF.rec_to_dict_list(sampleset.record,GraphP)
    best_sol = sols[0][0]
    
    for i in range(n_iter):
        # Update penalty
        print('sols[0][0][H_con]= ',sols[0][0]['H_con'])
        if sols[0][0]['H_con'] <= (5-n) and sols[0][0]['H_con']>-n:
            pen  = scaling[0]*pen # reduce pen if valid solution or less than 5 invalidities
        elif sols[0][0]['H_con'] == -n:
            pen = scaling[1]*pen
        elif sols[0][0]['H_con'] > (10-n):
            pen = scaling[2]*pen
        else:
            pen = scaling[3]*pen
            #print(best_sol['H_con'])
        print('new pen= ',pen,'in step ',i+1)

        # try to quantum solve and process the info
        try:
            if hybrid:
                sampleset = GraphP.hybrid_solve(pen)
            else:
                sampleset = GraphP.quantum_solve(pen)
            sols = HF.rec_to_dict_list(sampleset.record,GraphP)
        except:
            print('TimeoutError, check conexion')
            return best_sol,best_pen

        #try to correct n_correct sols
        fixed_sols = HF.fix_multiple_results(GraphP,sols,n_sols=n_fix,check=check)
        best_sol_loop = fixed_sols[0]

        # update best sol if better than previous and valid
        if best_sol_loop['H_con'] < best_sol['H_con'] or (best_sol_loop['H_con'] == best_sol['H_con'] and best_sol_loop['mod'] > best_sol['mod']):
            best_sol = best_sol_loop
            best_pen = pen
    return best_sol,best_pen