import numpy as np
from sklearn.preprocessing import StandardScaler
from simtk.openmm.app import PDBFile
from simtk.openmm.app import ForceField, Simulation
from simtk.openmm import LangevinIntegrator
from simtk.unit import kelvin, picoseconds, femtoseconds


def clean_data(df):
    """
    Clean the protein data by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): The input protein data.
    
    Returns:
    pd.DataFrame: The cleaned protein data.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def normalize_data(df):
    """
    Normalize the protein data using standard scaling.
    
    Parameters:
    df (pd.DataFrame): The input protein data.
    
    Returns:
    pd.DataFrame: The normalized protein data.
    """
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


def transform_data(df):
    """
    Transform the protein data for model input.
    
    Parameters:
    df (pd.DataFrame): The input protein data.
    
    Returns:
    pd.DataFrame: The transformed protein data.
    """
    # Example transformation: log transformation
    df = df.applymap(lambda x: np.log(x + 1))
    return df


def preprocess_protein_data(df):
    """
    Preprocess the protein data by cleaning, normalizing, and transforming it.
    
    Parameters:
    df (pd.DataFrame): The input protein data.
    
    Returns:
    pd.DataFrame: The preprocessed protein data.
    """
    df = clean_data(df)
    df = normalize_data(df)
    df = transform_data(df)
    return df


def parse_pdb_file(file_path):
    """
    Parse a PDB file and return the PDB object.
    
    Parameters:
    file_path (str): The path to the PDB file.
    
    Returns:
    PDBFile: The parsed PDB object.
    """
    pdb = PDBFile(file_path)
    return pdb


def create_simulation(pdb):
    """
    Create a simulation using the PDB object.
    
    Parameters:
    pdb (PDBFile): The PDB object.
    
    Returns:
    Simulation: The created simulation object.
    """
    forcefield = ForceField('amber99sb.xml')
    system = forcefield.createSystem(pdb.topology)
    integrator = LangevinIntegrator(300*kelvin, 1/picoseconds, 2*femtoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    return simulation
