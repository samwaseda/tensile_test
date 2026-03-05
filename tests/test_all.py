import unittest

from damask_local import main as dsk


class TestFull(unittest.TestCase):

    def test_get_lattice_structure(self):
        self.assertRaises(ValueError, dsk._get_lattice_structure)
        self.assertEqual(dsk._get_lattice_structure(chemical_symbol="Al"), "cF")
        self.assertEqual(dsk._get_lattice_structure(chemical_symbol="Aluminum"), "cF")
        self.assertEqual(dsk._get_lattice_structure(lattice="fcc"), "cF")
        self.assertEqual(dsk._get_lattice_structure(key="Hooke_Al"), "cF")
        self.assertEqual(
            dsk._get_lattice_structure(key="Hooke_Fe", lattice="fcc"), "cF"
        )
        self.assertEqual(
            dsk._get_lattice_structure(
                key="Hooke_Fe", chemical_symbol="Mg", lattice="fcc"
            ),
            "cF",
        )
        self.assertEqual(
            dsk._get_lattice_structure(key="Hooke_Fe", chemical_symbol="Mg"), "cI"
        )
        self.assertEqual(dsk._get_lattice_structure(chemical_symbol="Mg"), "hP")


if __name__ == "__main__":
    unittest.main()
