import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as Calculator


def molecular_descriptors(data):
    	#Setting up for molecular descriptors
    	n = data.shape[0]
    	list_of_descriptors = ['NumHeteroatoms', 'ExactMolWt', 'NOCount', 'NumHDonors',
        'RingCount', 'NumAromaticRings', 'NumSaturatedRings','NumAliphaticRings']
    	calc = Calculator(list_of_descriptors)
    	D = len(list_of_descriptors)
    	d = len(list_of_descriptors)*2 + 4

    	Y = data['EC_value']
    	X = np.zeros((n,d))
    	X[:,-3] = data['T']
    	X[:,-2] = data['P']
    	X[:,-1] = data['MOLFRC_A']

    	for i in range(n):
        	A = Chem.MolFromSmiles(data['A'][i])
        	B = Chem.MolFromSmiles(data['B'][i])
        	X[i][:D] = calc.CalcDescriptors(A)
        	X[i][D:2*D] = calc.CalcDescriptors(B)

    	new_data = pd.DataFrame(X,columns=['NUM', 'NumHeteroatoms_A', 'MolWt_A', 'NOCount_A',
        	'NumHDonors_A', 'RingCount_A', 'NumAromaticRings_A', 'NumSaturatedRings_A',
        	'NumAliphaticRings_A', 'NumHeteroatoms_B', 'MolWt_B', 'NOCount_B', 'NumHDonors_B',
        	'RingCount_B', 'NumAromaticRings_B', 'NumSaturatedRings_B', 'NumAliphaticRings_B',
        	'T', 'P', 'MOLFRC_A'])

		data_scaled = StandardScaler().fit_transform(new_data)
    	return data_scaled, Y

