import unittest
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.virtual_screening import select_compounds, dock_compounds, score_compounds, virtual_screening


class TestVirtualScreening(unittest.TestCase):


    def setUp(self):
        self.protein = Chem.MolFromPDBBlock("""
ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1      12.560  13.207  10.000  1.00  0.00           C  
ATOM      3  C   ALA A   1      13.104  14.207  11.000  1.00  0.00           C  
ATOM      4  O   ALA A   1      12.560  15.207  11.000  1.00  0.00           O  
ATOM      5  CB  ALA A   1      13.104  12.207  11.000  1.00  0.00           C  
""")
        self.compounds = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CCN'), Chem.MolFromSmiles('CCC')]
        self.criteria = {
            'MW': lambda x: Descriptors.MolWt(x) < 100,
            'LogP': lambda x: Descriptors.MolLogP(x) < 1
        }


    def test_select_compounds(self):
        selected_compounds = select_compounds(self.compounds, self.criteria)
        self.assertEqual(len(selected_compounds), 2, 
                         "Incorrect number of compounds selected")


    def test_dock_compounds(self):
        docking_scores = dock_compounds(self.protein, self.compounds)
        self.assertEqual(len(docking_scores), len(self.compounds), 
                         "Docking scores length mismatch")
        self.assertTrue(all(isinstance(score, float) for score in docking_scores), 
                        "Docking scores are not all floats")


    def test_score_compounds(self):
        docking_scores = [0.5, 0.8, 0.3]
        scored_compounds = score_compounds(self.compounds, docking_scores)
        self.assertEqual(len(scored_compounds), len(self.compounds), 
                         "Scored compounds length mismatch")
        self.assertTrue(all(isinstance(score, float) for _, score in scored_compounds), 
                        "Scores are not all floats")


    def test_virtual_screening(self):
        scored_compounds = virtual_screening(self.protein, self.compounds, self.criteria)
        self.assertEqual(len(scored_compounds), 2, 
                         "Incorrect number of scored compounds")
        self.assertTrue(all(isinstance(score, float) for _, score in scored_compounds), 
                        "Scores are not all floats")


if __name__ == '__main__':
    unittest.main()
