"""
Test to verify the DPI parameter in targetviz_report
"""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from targetviz.main import targetviz_report


class TestDPI(unittest.TestCase):
    """
    Test class for verifying the DPI parameter in targetviz_report
    """

    @patch("matplotlib.pyplot.savefig")
    def test_dpi_parameter(self, mock_savefig):
        """
        Test that the DPI parameter is correctly passed to the savefig function
        """
        # Create a simple test dataframe
        df = pd.DataFrame(
            {
                "target": np.random.normal(0, 1, 100),
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with default DPI (100)
            targetviz_report(df, "target", columns=["feature1"], output_dir=temp_dir + "/")

            # Check that savefig was called with dpi=100 (default)
            # We can't directly check the exact call arguments because savefig gets called
            # multiple times with different parameters, but we can verify that at least one
            # call had dpi=100
            dpi_values = [call_args[1].get("dpi") for call_args in mock_savefig.call_args_list]
            self.assertIn(100, dpi_values)

            # Reset the mock
            mock_savefig.reset_mock()

            # Test with custom DPI (300)
            custom_dpi = 300
            targetviz_report(
                df, "target", columns=["feature1"], dpi=custom_dpi, output_dir=temp_dir + "/"
            )

            # Check that savefig was called with the custom DPI value
            dpi_values = [call_args[1].get("dpi") for call_args in mock_savefig.call_args_list]
            self.assertIn(custom_dpi, dpi_values)
            self.assertNotIn(100, dpi_values)  # Ensure the default is not used
