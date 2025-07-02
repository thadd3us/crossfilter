// Crossfilter frontend application

// FilterOperationType enum to match backend schema.py
const FilterOperationType = {
    SPATIAL: 'spatial',
    TEMPORAL: 'temporal',
    RESET: 'reset'
};

class CrossfilterApp {
    constructor() {
        this.selectedRowIndices = new Set();
        this.plotData = null;
        this.hasData = false;
        this.initialize();
    }

    async initialize() {
        await this.checkSessionStatus();
        this.setupEventListeners();
    }

    async checkSessionStatus() {
        try {
            const response = await fetch('/api/session');
            const status = await response.json();
            this.hasData = status.has_data;
            this.updateStatus(status);
        } catch (error) {
            this.showError('Failed to check session status: ' + error.message);
        }
    }

    updateStatus(status) {
        const statusElement = document.getElementById('status');
        if (status.has_data) {
            statusElement.innerHTML = `
                <strong>Status:</strong> Data loaded - ${status.row_count} rows, 
                ${status.filtered_count} after filtering (${status.columns.length} columns)
            `;
            document.getElementById('refreshBtn').disabled = false;
            document.getElementById('resetFiltersBtn').disabled = false;
        } else {
            statusElement.innerHTML = '<strong>Status:</strong> No data loaded';
            document.getElementById('refreshBtn').disabled = true;
            document.getElementById('resetFiltersBtn').disabled = true;
        }
    }

    setupEventListeners() {
        // Plot selection handling will be set up when plot is created
    }

    async loadSampleData() {
        try {
            this.showInfo('Loading sample data...');
            const response = await fetch('/api/data/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    file_path: '/user_home/workspace/test_data/sample_100.jsonl'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.hasData = true;
            this.updateStatus(result.session_state);
            this.showInfo('Sample data loaded successfully. Loading temporal plot...');
            await this.refreshPlot();
        } catch (error) {
            this.showError('Failed to load sample data: ' + error.message);
        }
    }

    async refreshPlot() {
        if (!this.hasData) {
            this.showError('No data loaded');
            return;
        }

        try {
            this.showInfo('Loading temporal CDF plot...');
            const response = await fetch('/api/plots/temporal?max_groups=10000');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.plotData = result;
            this.renderTemporalCDF(result);
            this.clearMessages();
        } catch (error) {
            this.showError('Failed to load plot data: ' + error.message);
        }
    }

    renderTemporalCDF(plotData) {
        const plotContainer = document.getElementById('plotContainer');
        
        // Create the plot using Plotly
        const figure = plotData.plotly_plot;
        
        // Add selection handling to the plot
        const layout = {
            ...figure.layout,
            title: 'Temporal Distribution (CDF)',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Cumulative Probability' },
            hovermode: 'closest',
            selectdirection: 'horizontal'
        };

        // Enable selection on the plot
        const config = {
            displayModeBar: true,
            modeBarButtonsToAdd: ['select2d', 'lasso2d'],
            modeBarButtonsToRemove: ['autoScale2d'],
            displaylogo: false
        };

        Plotly.newPlot(plotContainer, figure.data, layout, config);

        // Handle plot selection events
        plotContainer.on('plotly_selected', (eventData) => {
            this.handlePlotSelection(eventData);
        });

        plotContainer.on('plotly_deselect', () => {
            this.clearSelection();
        });

        // Show plot controls now that plot is rendered
        document.getElementById('plotControls').style.display = 'flex';
        this.updateFilterButton();
    }

    handlePlotSelection(eventData) {
        if (!eventData || !eventData.points) {
            return;
        }

        // Extract row indices from selected points
        // The backend should provide df_id (row index) in the customdata
        const selectedIndices = new Set();
        eventData.points.forEach(point => {
            if (point.customdata && point.customdata.df_id !== undefined) {
                selectedIndices.add(point.customdata.df_id);
            }
        });

        this.selectedRowIndices = selectedIndices;
        this.updateFilterButton();
    }

    clearSelection() {
        this.selectedRowIndices.clear();
        this.updateFilterButton();
    }

    async resetFilters() {
        try {
            const response = await fetch('/api/filters/reset', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.clearSelection();
            this.showInfo('Filters reset successfully');
            await this.checkSessionStatus();
            await this.refreshPlot();
        } catch (error) {
            this.showError('Failed to reset filters: ' + error.message);
        }
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showInfo(message) {
        this.showMessage(message, 'info');
    }

    showMessage(message, type = 'info') {
        const messagesContainer = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = type;
        messageDiv.textContent = message;
        
        messagesContainer.innerHTML = '';
        messagesContainer.appendChild(messageDiv);
        
        // Auto-clear info messages after 3 seconds
        if (type === 'info') {
            setTimeout(() => {
                if (messagesContainer.contains(messageDiv)) {
                    messagesContainer.removeChild(messageDiv);
                }
            }, 3000);
        }
    }

    clearMessages() {
        document.getElementById('messages').innerHTML = '';
    }

    updateFilterButton() {
        const filterButton = document.getElementById('filterToVisibleBtn');
        const plotSelectionInfo = document.getElementById('plotSelectionInfo');
        
        if (this.selectedRowIndices.size > 0) {
            filterButton.disabled = false;
            plotSelectionInfo.textContent = `Selected: ${this.selectedRowIndices.size} points`;
        } else {
            filterButton.disabled = true;
            plotSelectionInfo.textContent = '';
        }
    }

    async filterToSelected(eventSource) {
        const selectedIndices = Array.from(this.selectedRowIndices);
        
        if (selectedIndices.length === 0) {
            this.showError('No points selected. Use lasso or box select to choose points first.');
            return;
        }

        // Get display names for different event sources
        const displayNames = {
            [FilterOperationType.TEMPORAL]: 'temporal buckets',
            [FilterOperationType.SPATIAL]: 'spatial regions'
        };
        
        const displayName = displayNames[eventSource] || 'points';

        try {
            this.showInfo(`Filtering to ${selectedIndices.length} selected ${displayName}...`);
            
            const response = await fetch('/api/filters/df_ids', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    df_ids: selectedIndices,
                    event_source: eventSource,
                    description: `Filter to ${selectedIndices.length} selected ${displayName} from ${eventSource} plot`
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showInfo(`Filtered to selected ${displayName}: ${result.filtered_count} rows remaining`);
            
            // Refresh session status and plot
            await this.checkSessionStatus();
            await this.refreshPlot();
        } catch (error) {
            this.showError(`Failed to apply ${eventSource} selection filter: ` + error.message);
        }
    }

    async filterTemporalToSelected() {
        return this.filterToSelected(FilterOperationType.TEMPORAL);
    }
}

// Global functions for HTML onclick handlers
let app;

function loadSampleData() {
    app.loadSampleData();
}

function refreshPlot() {
    app.refreshPlot();
}

function resetFilters() {
    app.resetFilters();
}

function filterTemporalToSelected() {
    app.filterTemporalToSelected();
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    app = new CrossfilterApp();
});