"""Unit tests for generate_artificial_data module."""

import unittest

from training_pipeline import generate_artificial_data


class TestGenerateArtificialData(unittest.TestCase):
    """Unit tests for generate_artificial_data module."""

    def test_sigmoid(self) -> None:
        """Tests sigmoid() function."""
        self.assertAlmostEqual(0.5, generate_artificial_data.sigmoid(0.0), delta=0.0)

        self.assertAlmostEqual(0.0, generate_artificial_data.sigmoid(-50.0), delta=1e-20)
        self.assertAlmostEqual(1.0, generate_artificial_data.sigmoid(+50.0), delta=1e-20)

        self.assertAlmostEqual(0.0, generate_artificial_data.sigmoid(-1000.0), delta=0.0)
        self.assertAlmostEqual(1.0, generate_artificial_data.sigmoid(+1000.0), delta=0.0)

        self.assertAlmostEqual(
            0.26894142136999512074,
            generate_artificial_data.sigmoid(-1.0),
            delta=0.0,
        )
        self.assertAlmostEqual(
            0.73105857863000487925,
            generate_artificial_data.sigmoid(1.0),
            delta=0.0,
        )

    def test_generate_data(self) -> None:
        """Tests generate_data() function."""
        data = generate_artificial_data.generate_data(100, 10, 12345)
        self.assertEqual((100, 11), data.shape)


if __name__ == "__main__":
    unittest.main()
