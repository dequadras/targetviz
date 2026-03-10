import os
import tempfile
import zipfile

import pandas as pd

from targetviz.profile_report import targetviz_report


def test_html_and_zip_output():
    """Test that reports can be generated as both HTML and zipped HTML files."""
    # Create a simple test dataframe
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "a", "b"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test regular HTML output
        html_filename = "report.html"
        full_html_path = os.path.join(temp_dir, html_filename)

        targetviz_report(
            data=data, target="target", output_dir=temp_dir + "/", name_file_out=html_filename
        )

        # Check that HTML file was created
        assert os.path.exists(full_html_path)
        with open(full_html_path, "r") as f:
            html_content = f.read()
            assert "<html" in html_content
            assert "<body" in html_content

        # Test zip output
        zip_filename = "report.html.zip"
        full_zip_path = os.path.join(temp_dir, zip_filename)

        targetviz_report(
            data=data, target="target", output_dir=temp_dir + "/", name_file_out=zip_filename
        )

        # Check that zip file was created
        assert os.path.exists(full_zip_path)
        assert not os.path.exists(os.path.join(temp_dir, "report.html"))

        # Verify zip file contains HTML file
        with zipfile.ZipFile(full_zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            assert "report.html" in file_list

            # Extract and check HTML content
            zip_ref.extract("report.html", temp_dir)
            extracted_path = os.path.join(temp_dir, "report.html")
            with open(extracted_path, "r") as f:
                extracted_content = f.read()
                assert "<html" in extracted_content
                assert "<body" in extracted_content
