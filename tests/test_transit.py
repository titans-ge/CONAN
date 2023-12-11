import unittest

class TestTransitModel(unittest.TestCase):
    def setUp(self):
        # Initialize the Transit_Model object with sample parameters
        T0 = [0.0]
        RpRs = [0.1]
        b = [0.5]
        dur = [2.0]
        per = [10.0]
        eos = [0.0]
        eoc = [0.0]
        ddf = [0.0]
        occ = [0.0]
        c1 = [0.0]
        c2 = [0.0]
        npl = 1
        self.model = Transit_Model(T0, RpRs, b, dur, per, eos, eoc, ddf, occ, c1, c2, npl)

    def test_get_value(self):
        # Test the get_value method with a sample time array
        tarr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        expected_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Replace with expected values
        values = self.model.get_value(tarr)
        self.assertEqual(values, expected_values)

    def test_parameter_names(self):
        # Test the parameter_names attribute
        expected_names = ['T0', 'RpRs', 'b', 'dur', 'per', 'eos', 'eoc', 'ddf', 'occ', 'c1', 'c2']
        names = self.model.parameter_names
        self.assertEqual(names, expected_names)

if __name__ == '__main__':
    unittest.main()