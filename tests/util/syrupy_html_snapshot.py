from syrupy.extensions.single_file import SingleFileSnapshotExtension


class HTMLSnapshotExtension(SingleFileSnapshotExtension):
    """Custom syrupy extension to save HTML files with .html extension."""

    _file_extension = "html"

    def serialize(self, data: str | bytes, **kwargs) -> bytes:
        """Serialize string data to bytes for file storage."""
        if isinstance(data, str):
            return data.encode("utf-8")
        return super().serialize(data, **kwargs)
