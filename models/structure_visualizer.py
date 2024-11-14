import torch
import py3Dmol
from typing import Dict, List, Optional, Union
from biotite.structure import AtomArray
from biotite.structure.io import pdb
from biotite.structure.residues import get_residue_positions

class ProteinStructureVisualizer:
    """Real-time protein structure visualization with interactive features."""

    def __init__(self, width: int = 800, height: int = 600):
        """Initialize visualizer with display dimensions."""
        self.width = width
        self.height = height
        self.current_structure = None
        self.view = None

    def initialize_viewer(self) -> py3Dmol.view:
        """Initialize the 3D viewer."""
        self.view = py3Dmol.view(width=self.width, height=self.height)
        return self.view

    def load_structure(self, pdb_data: Union[str, AtomArray]) -> None:
        """Load protein structure from PDB data or AtomArray."""
        if isinstance(pdb_data, str):
            self.current_structure = pdb.PDBFile.read(pdb_data)
        else:
            self.current_structure = pdb_data

        # Convert structure to PDB format for py3Dmol
        pdb_string = pdb.PDBFile.write(self.current_structure)
        self.view.addModel(pdb_string, "pdb")

    def highlight_residues(
        self,
        residue_indices: List[int],
        color: str = "red",
        style: Dict = {"stick": {}}
    ) -> None:
        """Highlight specific residues in the structure."""
        selection = {"resi": residue_indices}
        self.view.setStyle(selection, style)
        self.view.setColor(selection, color)

    def show_surface(
        self,
        opacity: float = 0.8,
        color_scheme: str = "spectrum"
    ) -> None:
        """Display protein surface with customizable appearance."""
        self.view.addSurface(py3Dmol.VDW, {
            "opacity": opacity,
            "colorscheme": color_scheme
        })

    def highlight_domains(
        self,
        domain_ranges: List[Dict[str, Union[int, str]]],
        colors: Optional[List[str]] = None
    ) -> None:
        """Highlight protein domains with different colors."""
        if colors is None:
            colors = ["red", "blue", "green", "yellow", "purple", "orange"]

        for i, domain in enumerate(domain_ranges):
            color = colors[i % len(colors)]
            selection = {
                "resi": list(range(domain["start"], domain["end"] + 1))
            }
            self.view.setStyle(selection, {"cartoon": {"color": color}})

    def show_interactions(
        self,
        interaction_pairs: List[Dict[str, Union[int, float]]],
        interaction_type: str = "hbond"
    ) -> None:
        """Visualize protein interactions (H-bonds, salt bridges, etc.)."""
        for interaction in interaction_pairs:
            start = interaction["residue1"]
            end = interaction["residue2"]
            strength = interaction.get("strength", 1.0)

            self.view.addCylinder({
                "start": {"resi": start},
                "end": {"resi": end},
                "radius": 0.1 * strength,
                "color": "yellow" if interaction_type == "hbond" else "red",
                "dashed": True
            })

    def highlight_active_site(
        self,
        active_site_residues: List[int],
        style: Dict = {"stick": {}, "sphere": {"radius": 0.5}}
    ) -> None:
        """Highlight active site residues with custom visualization."""
        selection = {"resi": active_site_residues}
        self.view.setStyle(selection, style)
        self.view.addSurface(py3Dmol.VDW, {
            "opacity": 0.6,
            "colorscheme": "yellowCarbon"
        }, {"resi": active_site_residues})

    def add_labels(
        self,
        residue_labels: Dict[int, str],
        font_size: int = 12,
        color: str = "black"
    ) -> None:
        """Add custom labels to specific residues."""
        for residue_id, label_text in residue_labels.items():
            self.view.addLabel(label_text, {
                "position": {"resi": residue_id},
                "fontSize": font_size,
                "color": color,
                "backgroundColor": "white",
                "backgroundOpacity": 0.5
            })

    def create_animation(
        self,
        frames: List[AtomArray],
        frame_delay: int = 500
    ) -> None:
        """Create animation from multiple structure frames."""
        for i, frame in enumerate(frames):
            pdb_string = pdb.PDBFile.write(frame)
            model_number = i + 1
            self.view.addModel(pdb_string, "pdb", {"model": model_number})

        self.view.animate({"loop": "forward", "reps": 0, "interval": frame_delay})

    def set_camera(
        self,
        zoom: float = 2.0,
        rotation: Dict[str, float] = {"x": 0, "y": 0, "z": 0}
    ) -> None:
        """Set camera position and orientation."""
        self.view.zoomTo()
        self.view.zoom(zoom)
        self.view.rotate(rotation["x"], "x")
        self.view.rotate(rotation["y"], "y")
        self.view.rotate(rotation["z"], "z")

    def save_image(self, filename: str, width: int = 1024, height: int = 1024) -> None:
        """Save current view as high-resolution image."""
        self.view.resize(width, height)
        self.view.png(filename)
        self.view.resize(self.width, self.height)  # Reset to original size

    def clear(self) -> None:
        """Clear all visualizations."""
        if self.view is not None:
            self.view.clear()
            self.current_structure = None
