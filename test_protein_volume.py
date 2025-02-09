# o-3-mini-high

import unittest
import math

from protein_volume import compute_protein_volume

class TestProteinVolume(unittest.TestCase):
    def test_no_atoms(self):
        """Test that an empty list of atoms gives volume 0."""
        atoms = []
        vol = compute_protein_volume(atoms, n_div=50)
        self.assertEqual(vol, 0.0)

    def test_single_sphere(self):
        """Test a single sphere (atom) centered at (0,0,0) with radius 1."""
        atoms = [{
            'atom_type': 'O',
            'aa': 'ALA',
            'residue_number': 1,
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'charge': 0.0,
            'radius': 1.0
        }]
        vol = compute_protein_volume(atoms, n_div=100)
        expected = (4.0/3.0) * math.pi * (1.0**3)
        rel_error = abs(vol - expected) / expected
        print("Single sphere: approximated volume =", vol, "exact volume =", expected,
              "relative error =", rel_error)
        self.assertLessEqual(rel_error, 0.1)

    def test_two_nonoverlapping_spheres(self):
        """Test two non-overlapping spheres (each of radius 1)."""
        atoms = [
            {'atom_type': 'O', 'aa': 'ALA', 'residue_number': 1,
             'x': -3.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0, 'radius': 1.0},
            {'atom_type': 'O', 'aa': 'ALA', 'residue_number': 2,
             'x': 3.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0, 'radius': 1.0}
        ]
        vol = compute_protein_volume(atoms, n_div=100)
        expected = 2 * ((4.0/3.0) * math.pi * (1.0**3))
        rel_error = abs(vol - expected) / expected
        print("Two non-overlapping spheres: approximated volume =", vol, "exact volume =", expected,
              "relative error =", rel_error)
        self.assertLessEqual(rel_error, 0.1)

    def test_overlapping_spheres(self):
        """Test two completely overlapping spheres (identical centers and radii).
           The union should equal one sphere's volume."""
        atoms = [
            {'atom_type': 'O', 'aa': 'ALA', 'residue_number': 1,
             'x': 0.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0, 'radius': 1.0},
            {'atom_type': 'O', 'aa': 'ALA', 'residue_number': 2,
             'x': 0.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0, 'radius': 1.0}
        ]
        vol = compute_protein_volume(atoms, n_div=100)
        expected = (4.0/3.0) * math.pi * (1.0**3)
        rel_error = abs(vol - expected) / expected
        print("Overlapping spheres: approximated volume =", vol, "exact volume =", expected,
              "relative error =", rel_error)
        self.assertLessEqual(rel_error, 0.1)

if __name__ == '__main__':
    unittest.main()
