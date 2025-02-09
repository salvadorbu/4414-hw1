# o-3-mini-high

import numpy as np
import argparse

def parse_pqr(filename):
    """
    Parse a PQR file into a list of atom dictionaries.
    
    The file is expected to contain lines starting with "ATOM". For each ATOM line, 
    the fields are assumed to be (in order):
      1. Record type ("ATOM")
      2. Atom serial number (ignored in this parser)
      3. Atom type (first letter is the chemical element, e.g. "HB3" is a hydrogen)
      4. Amino-acid name
      5. Residue number
      6. x coordinate
      7. y coordinate
      8. z coordinate
      9. Charge
      10. Radius
    Any REMARK or other lines are ignored.
    
    Returns:
        List[dict]: A list of dictionaries, one per atom.
    """
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip non-ATOM lines (e.g., REMARKs)
            if not line.startswith("ATOM"):
                continue

            # Split the line on whitespace
            parts = line.split()

            # Check that we have enough parts
            if len(parts) < 10:
                continue

            try:
                atom = {
                    "atom_type": parts[2],              # Field 3: atom type
                    "aa": parts[3],                     # Field 4: amino-acid name
                    "residue_number": int(parts[4]),      # Field 5: residue number
                    "x": float(parts[5]),               # Field 6: x coordinate
                    "y": float(parts[6]),               # Field 7: y coordinate
                    "z": float(parts[7]),               # Field 8: z coordinate
                    "charge": float(parts[8]),          # Field 9: charge
                    "radius": float(parts[9]),          # Field 10: radius
                }
            except ValueError as e:
                # Handle possible conversion errors gracefully
                print(f"Error parsing line: {line.strip()}\nError: {e}")
                continue

            atoms.append(atom)
    return atoms


def compute_protein_volume(atoms, n_div=100):
    """
    Compute the approximate volume of the protein using a voxel method.
    
    The idea is to enclose the protein in a box, partition the box into voxels,
    and count the fraction whose centers lie within at least one atom (considered
    as a sphere). The approximate protein volume is then given by:
    
        Vp ≈ (volume of box) * (number of inside voxels) / (total voxels)
    
    Args:
        atoms (list): List of atom dictionaries.
        n_div (int): Number of subdivisions along each axis (default 100).
        
    Returns:
        float: Approximate protein volume.
    """
    if not atoms:
        return 0.0

    # Compute the bounding box. Because each atom is a sphere, extend the box
    # by each atom's radius.
    x_min = min(atom['x'] - atom['radius'] for atom in atoms)
    x_max = max(atom['x'] + atom['radius'] for atom in atoms)
    y_min = min(atom['y'] - atom['radius'] for atom in atoms)
    y_max = max(atom['y'] + atom['radius'] for atom in atoms)
    z_min = min(atom['z'] - atom['radius'] for atom in atoms)
    z_max = max(atom['z'] + atom['radius'] for atom in atoms)

    a = x_max - x_min
    b = y_max - y_min
    c = z_max - z_min

    # Determine the grid spacing in each dimension.
    dx = a / n_div
    dy = b / n_div
    dz = c / n_div

    # Generate the voxel center coordinates.
    # We use linspace so that the first center is dx/2 from the minimum, etc.
    x_centers = np.linspace(x_min + dx/2, x_max - dx/2, n_div)
    y_centers = np.linspace(y_min + dy/2, y_max - dy/2, n_div)
    z_centers = np.linspace(z_min + dz/2, z_max - dz/2, n_div)
    
    # Create a 3D grid of voxel centers.
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    # Flatten the grid into a list of voxel center coordinates.
    voxel_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    N_voxels = voxel_coords.shape[0]

    # Create a boolean array that will be True for voxels inside the protein.
    inside = np.zeros(N_voxels, dtype=bool)

    # For each atom, mark voxels that are within the atom's sphere.
    # (We use vectorized numpy operations for speed.)
    for atom in atoms:
        center = np.array([atom['x'], atom['y'], atom['z']])
        r_sq = atom['radius'] ** 2
        # Compute squared distances from the voxel centers to the atom center.
        dist_sq = np.sum((voxel_coords - center)**2, axis=1)
        # Mark voxels where the voxel center lies within the sphere.
        inside |= (dist_sq <= r_sq)

    n_inside = np.count_nonzero(inside)
    volume_box = a * b * c
    protein_volume = volume_box * (n_inside / N_voxels)
    return protein_volume

def main():
    parser = argparse.ArgumentParser(
        description='Approximate protein volume using a voxel-based algorithm.')
    parser.add_argument('pqr_file', type=str, help='Path to the PQR file.')
    parser.add_argument('--n_div', type=int, default=100,
                        help=('Number of subdivisions per axis (default 100). '
                              'A higher number gives a finer grid and more accuracy.'))
    args = parser.parse_args()

    atoms = parse_pqr(args.pqr_file)
    if not atoms:
        print("No atoms were found in the file.")
        return

    vol = compute_protein_volume(atoms, n_div=args.n_div)
    print(f"Approximated protein volume: {vol:.3f} (cubic units)")

if __name__ == '__main__':
    main()
