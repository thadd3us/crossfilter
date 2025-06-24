"""Session state management for single-session Crossfilter application."""

import pandas as pd


class SessionState:
    """
    Manages the current state of data loaded into the Crossfilter application.
    
    This class represents the single session state for the application, holding
    the currently loaded dataset and any derived data structures needed for
    crossfiltering and visualization.
    
    In the single-session design pattern, there is exactly one instance of this
    class per web server instance, simplifying state management and eliminating
    the need for complex multi-user session handling.
    """
    
    def __init__(self) -> None:
        """Initialize session state with empty DataFrame."""
        self._data = pd.DataFrame()
        self._update_metadata()
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the current dataset."""
        return self._data
    
    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        """Set the current dataset."""
        self._data = value
        self._update_metadata()
    
    def _update_metadata(self) -> None:
        """Update metadata based on current DataFrame."""
        self._metadata = {
            "shape": self._data.shape,
            "columns": list(self._data.columns),
            "dtypes": self._data.dtypes.to_dict(),
        }
    
    @property
    def metadata(self) -> dict:
        """Get metadata about the current dataset."""
        return self._metadata.copy()
    
    def has_data(self) -> bool:
        """Check if the session has data loaded."""
        return not self._data.empty
    
    def clear(self) -> None:
        """Clear all data from the session state."""
        self._data = pd.DataFrame()
        self._update_metadata()
    
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the session state."""
        self.data = df
    
    def get_summary(self) -> dict:
        """Get a summary of the current session state."""
        if not self.has_data():
            return {"status": "empty", "message": "No data loaded"}
        
        return {
            "status": "loaded",
            "shape": self._metadata["shape"],
            "columns": self._metadata["columns"],
            "memory_usage": f"{self._data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }