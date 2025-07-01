// Crossfilter frontend application
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
        this.updateSelectionInfo();
        
        // Apply the filter to backend
        if (selectedIndices.size > 0) {
            this.applyTemporalFilter(Array.from(selectedIndices));
        }
    }

    async applyTemporalFilter(rowIndices) {
        try {
            const response = await fetch('/api/filters/apply', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    row_indices: rowIndices,
                    operation_type: 'temporal',
                    description: `Selected ${rowIndices.length} points from temporal CDF`
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showInfo(`Applied temporal filter: ${rowIndices.length} points selected`);
            
            // Refresh session status to show updated filter counts
            await this.checkSessionStatus();
        } catch (error) {
            this.showError('Failed to apply temporal filter: ' + error.message);
        }
    }

    clearSelection() {
        this.selectedRowIndices.clear();
        this.updateSelectionInfo();
    }

    updateSelectionInfo() {
        const selectionInfo = document.getElementById('selectionInfo');
        if (this.selectedRowIndices.size > 0) {
            selectionInfo.textContent = `Selected: ${this.selectedRowIndices.size} points`;
        } else {
            selectionInfo.textContent = '';
        }
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

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    app = new CrossfilterApp();
});