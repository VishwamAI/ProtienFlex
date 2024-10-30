# ProtienFlex

## Project Description

ProtienFlex is a project aimed at accelerating drug discovery by leveraging advanced techniques such as protein folding, virtual screening, and natural language processing (NLP). The primary goal is to reduce the time required for drug discovery and increase the effectiveness of identified compounds. This is achieved by using 3D models like AlphaFold in combination with human NLP and development.

## Purpose and Goals

The purpose of ProtienFlex is to streamline the drug discovery process by integrating various cutting-edge technologies. The main goals of the project are:
- To reduce the time required for drug discovery.
- To increase the effectiveness of identified drug compounds.
- To utilize 3D models like AlphaFold for accurate protein folding predictions.
- To employ NLP techniques for processing human language data relevant to drug discovery.

## Setup and Installation

To set up and run the ProtienFlex project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAI/ProtienFlex.git
   cd ProtienFlex
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```bash
   python src/main.py
   ```

## Contributing

We welcome contributions to the ProtienFlex project. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## Using OpenMM and PDB for Protein Development

ProtienFlex leverages OpenMM and PDB for protein development. OpenMM is a high-performance toolkit for molecular simulation, and PDB (Protein Data Bank) files contain 3D structures of proteins. Here are some steps to use OpenMM and PDB in the project:

1. Install OpenMM:
   ```bash
   conda install -c conda-forge openmm
   ```

2. Load a PDB file in OpenMM:
   ```python
   from simtk.openmm.app import PDBFile
   pdb = PDBFile('path_to_pdb_file.pdb')
   ```

3. Create a system and simulation:
   ```python
   from simtk.openmm.app import ForceField, Simulation
   from simtk.openmm import LangevinIntegrator
   from simtk.unit import kelvin, picoseconds, femtoseconds

   forcefield = ForceField('amber99sb.xml')
   system = forcefield.createSystem(pdb.topology)
   integrator = LangevinIntegrator(300*kelvin, 1/picoseconds, 2*femtoseconds)
   simulation = Simulation(pdb.topology, system, integrator)
   simulation.context.setPositions(pdb.positions)
   ```

4. Run the simulation:
   ```python
   simulation.minimizeEnergy()
   simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
   simulation.step(10000)
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
