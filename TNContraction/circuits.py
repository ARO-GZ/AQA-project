import GP.helper_functions as HF
import quimb as qu
import quimb.tensor as qtn

def Toffoli(circ,q1,q2,q3):
    '''
    Implement Toffoli gate on 3 qubits
    Args:
    q1,q2,q3 are the qubits on which the gate
    will act; q3 is the TARGET
    '''
    # build T dagger gate
    T_dagger_gate = qtn.circuit.build_gate_1(qu.T_gate().H, tags='T_dagger')

    # apply gates that equivalent to TOFFOLI
    circ.apply_gate('H',q3)
    circ.apply_gate('CNOT',q2,q3)
    T_dagger_gate(circ._psi,q3)
    circ.apply_gate('CNOT',q1,q3)
    circ.apply_gate('T',q3)
    circ.apply_gate('CNOT',q2,q3)
    T_dagger_gate(circ._psi,q3)
    circ.apply_gate('CNOT',q1,q3)
    circ.apply_gate('T',q2)
    circ.apply_gate('T',q3)
    circ.apply_gate('H',q3)
    circ.apply_gate('CNOT',q1,q2)
    circ.apply_gate('T',q1)
    T_dagger_gate(circ._psi,q2)
    circ.apply_gate('CNOT',q1,q2)

def build_toffoli_circuit():
        # Define circuit
    N = 3

    circ = qtn.Circuit(N=N)

    #initial layer of hadamards
    for i in range(N):
        circ.apply_gate('H', i, gate_round=0)
        circ.apply_gate('RZ', 1.234, i, gate_round=0)

    circ.apply_gate('CNOT', 0,  1, gate_round=1)

    for i in range(N):
        #circ.apply_gate('H', i, gate_round=2)
        circ.apply_gate('RZ', 1.234, i, gate_round=2)

    circ.apply_gate('CNOT', 1,  2, gate_round=1)

    for i in range(N):
        circ.apply_gate('H', i, gate_round=2)
        circ.apply_gate('RZ', 1.234, i, gate_round=2)

    Toffoli(circ,0,1,2)

    return circ

def build_toffoli_circuit_pos(pos: str ='end'):
    '''
    Creates 3-qubit Quantum Circuit in quimb
    with 1 toffoli, 2 CNOTs and several 1-qubit
    gates.
    Args: 
        - pos: position of the toffoli gate
            'start': begining of the circuit
            'middle': 
            'end':
    '''
        # Define circuit
    N = 3

    circ = qtn.Circuit(N=N)

    if pos == 'start':
        Toffoli(circ,0,1,2)

    #initial layer of hadamards
    for i in range(N):
        circ.apply_gate('H', i, gate_round=0)
        circ.apply_gate('RZ', 1.234, i, gate_round=0)

    circ.apply_gate('CNOT', 0,  1, gate_round=1)

    if pos == 'middle':
        Toffoli(circ,0,1,2)

    for i in range(N):
        #circ.apply_gate('H', i, gate_round=2)
        circ.apply_gate('RZ', 1.234, i, gate_round=2)

    circ.apply_gate('CNOT', 1,  2, gate_round=1)

    for i in range(N):
        circ.apply_gate('H', i, gate_round=2)
        circ.apply_gate('RZ', 1.234, i, gate_round=2)

    if pos == 'end':
        Toffoli(circ,0,1,2)

    return circ