import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector,DensityMatrix
from qiskit.visualization import plot_bloch_multivector




def get_operator_list(measurement_list):
    '''
    function that return matrix representation of measurement operators given a list of measurement_directions
    
    Arg: measurement_list , a list of state_vectors. (2 state_vectors per direction)
    Returns a list of operators
    '''
    op_list=[]
    for state in measurement_list:
        operator=np.outer(state,state)
        operator=operator/(len(measurement_list)/2)
        op_list.append(operator)
        
    return op_list

def spherical_to_amplitudes(measurement_basis):
    
    '''
    function that gives state_vectors when measurement directions are given in angles
    Arg: measurement_basis , a list of (theta,phi) tuples for directions
    
    Returns a list of state_vectors , i.e a measurement_list
    '''
    measurement_list=[]
    for basis in measurement_basis:
        a1=np.cos(basis[0]/2)
        b1=np.sin(basis[0]/2)*np.exp(1.0j*basis[1])
        a2=-np.sin([basis[0]/2])
        b2=np.cos([basis[0]/2])*np.exp(1.0j*basis[1])
        positive_direction=[a1,b1]
        negative_direction=[a2,b2]
        measurement_list.append(positive_direction)
        measurement_list.append(negative_direction)
    return measurement_list



def measure_results(measurement_basis,backend,initial_state=np.array([1,0])):
    '''
    function that gives the counts corresponding to all the measurements
    Arguments: measurement_basis, backend to use and the initial_state of the qubit.
    '''
    
    outcomes=[]
    for basis in measurement_basis:
        circuit=QuantumCircuit(1)
        circuit.initialize(initial_state)
        circuit.u(basis[0],basis[1],0,0)
        
        circuit.measure_all()
        transpiled_circuit = transpile(circuit, backend)
        
        transpiled_circuit.draw('mpl')
        plt.show()
        job = backend.run(transpiled_circuit)
        counts = job.result().get_counts()
        
        outcomes.append(counts)
        
    return outcomes

def get_normalized_outcomes(outcomes, weights=[0.5,0.5]):
    '''
    normalizes the outcomes , so that all results sum to one.
    
    '''
    
    normalized_outcomes=outcomes.copy()
    total_counts=0
    for i,counts in enumerate(normalized_outcomes):
        
        total_counts=total_counts+(sum(normalized_outcomes[i].values()))*weights[i]
      
        if len(counts)==1:
            normalized_outcomes[i]['dummy']=0.0
        
    
    for i in range(len(normalized_outcomes)):
        for key in normalized_outcomes[i]:
            normalized_outcomes[i][key]=normalized_outcomes[i][key]*weights[i]/total_counts
    
    
    
    return normalized_outcomes

def normalized_outcomes_to_frequency_vector(normalized_outcomes):
    '''
    Returns the final_frequency vector from normalized_outcomes
    
    '''
    
    dim=len(normalized_outcomes)*2
    f=np.zeros(dim)
    for i,outcomes in enumerate(normalized_outcomes):
        
        f[0+2*i]=outcomes.get('0',0.0)
        f[1+2*i]=outcomes.get('1',0.0)
        
    return f

def get_frequencies(measurement_basis,backend,initial_state=np.array([1,0]),weights=[0.5,0.5]):
    outcomes=measure_results(measurement_basis,backend,initial_state)
    normalized_outcomes=get_normalized_outcomes(outcomes,weights)
    f=normalized_outcomes_to_frequency_vector(normalized_outcomes)
    return f
