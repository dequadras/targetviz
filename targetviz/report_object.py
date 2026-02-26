"""TargetVizReport: wrapper that enables notebook-embedded display."""

import html
import os
import zipfile


class TargetVizReport:
    """Wraps a rendered targetviz HTML report.

    In a Jupyter notebook, simply evaluate the object in a cell to see the
    report inline (via ``_repr_html_``).  You can also save it to disk with
    :meth:`to_file` or retrieve the raw HTML string with :meth:`to_html`.
    """

    def __init__(self, html_content: str) -> None:
        self._html = html_content

    # ------------------------------------------------------------------
    # Notebook integration
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """Render the report inside an iframe in Jupyter notebooks.

        The full report is embedded via the ``srcdoc`` attribute so that the
        report's CSS / JS is completely isolated from the notebook.  This is
        the same strategy that ydata-profiling uses.
        """
        escaped = html.escape(self._html, quote=True)
        return (
            '<iframe width="100%" frameborder="0" '
            f'srcdoc="{escaped}" '
            'style="height: 80vh; border: 1px solid #ddd; border-radius: 4px;">'
            "</iframe>"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        """Return the raw HTML string."""
        return self._html

    def to_file(self, path: str) -> None:
        """Write the report to an HTML (or HTML-zipped) file.

        Parameters
        ----------
        path : str
            Destination file path.  If *path* ends with ``.html.zip`` the
            HTML is placed inside a ZIP archive.
        """
        if path.endswith(".html.zip"):
            html_filename = path[:-4]
            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(self._html)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(html_filename, os.path.basename(html_filename))
            os.remove(html_filename)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._html)
