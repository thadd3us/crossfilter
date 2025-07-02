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
        this.eventSource = null;
        this.filterVersion = 0;
        this.initialize();
    }

    async initialize() {
        console.log('CrossfilterApp: Initializing...');
        await this.checkSessionStatus();
        this.setupEventListeners();
        this.setupSSEConnection();
        console.log('CrossfilterApp: Initialization complete');
    }

    async checkSessionStatus() {
        try {
            console.log('CrossfilterApp: Checking session status...');
            const response = await fetch('/api/session');
            const status = await response.json();
            console.log('CrossfilterApp: Session status:', status);
            this.hasData = status.has_data;
            this.updateStatus(status);
            
            // Auto-load plot if data is already present
            if (this.hasData) {
                console.log('CrossfilterApp: Data found, loading plot...');
                await this.refreshPlot();
            } else {
                console.log('CrossfilterApp: No data found');
            }
        } catch (error) {
            this.showError('Failed to check session status: ' + error.message);
        }
    }

    updateStatus(status) {
        const statusElement = document.getElementById('status');
        const resetFiltersBtn = document.getElementById('resetFiltersBtn');
        
        if (status.has_data) {
            if (statusElement) {
                statusElement.innerHTML = `
                    <strong>Status:</strong> Data loaded - ${status.row_count} rows, 
                    ${status.filtered_count} after filtering (${status.columns.length} columns)
                `;
            }
            if (resetFiltersBtn) {
                resetFiltersBtn.disabled = false;
            }
        } else {
            if (statusElement) {
                statusElement.innerHTML = '<strong>Status:</strong> No data loaded';
            }
            if (resetFiltersBtn) {
                resetFiltersBtn.disabled = true;
            }
        }
    }

    setupEventListeners() {
        // Plot selection handling will be set up when plot is created
    }

    setupSSEConnection() {
        console.log('CrossfilterApp: Setting up SSE connection...');
        
        // Close existing connection if any
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Create new SSE connection
        this.eventSource = new EventSource('/api/events/filter-changes');
        
        this.eventSource.onopen = () => {
            console.log('CrossfilterApp: SSE connection opened');
        };
        
        this.eventSource.onmessage = (event) => {
            this.handleSSEEvent(event);
        };
        
        this.eventSource.onerror = (error) => {
            console.error('CrossfilterApp: SSE connection error:', error);
            this.showError('Lost connection to server. Refresh page to reconnect.');
        };
    }

    handleSSEEvent(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('CrossfilterApp: Received SSE event:', data);
            
            // Update filter version
            this.filterVersion = data.version;
            
            switch (data.type) {
                case 'connection_established':
                    console.log('CrossfilterApp: SSE connection established');
                    // Update status with current session state
                    if (data.session_state) {
                        this.updateStatus(data.session_state);
                    }
                    break;
                    
                case 'data_loaded':
                    console.log('CrossfilterApp: Data loaded event received');
                    this.hasData = data.session_state.has_data;
                    this.updateStatus(data.session_state);
                    if (this.hasData) {
                        this.refreshPlot();
                    }
                    break;
                    
                case 'filter_applied':
                    console.log('CrossfilterApp: Filter applied event received');
                    this.updateStatus(data.session_state);
                    this.refreshPlot();
                    break;
                    
                case 'filter_reset':
                    console.log('CrossfilterApp: Filter reset event received');
                    this.clearSelection();
                    this.updateStatus(data.session_state);
                    this.refreshPlot();
                    break;
                    
                case 'heartbeat':
                    // Silent heartbeat, just log debug info
                    console.debug('CrossfilterApp: SSE heartbeat');
                    break;
                    
                default:
                    console.warn('CrossfilterApp: Unknown SSE event type:', data.type);
            }
        } catch (error) {
            console.error('CrossfilterApp: Error parsing SSE event:', error);
        }
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
        console.log('CrossfilterApp: refreshPlot() called, hasData:', this.hasData);
        if (!this.hasData) {
            this.showError('No data loaded');
            return;
        }

        try {
            console.log('CrossfilterApp: Fetching plot data...');
            this.showInfo('Loading temporal CDF plot...');
            const response = await fetch('/api/plots/temporal?max_groups=10000');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('CrossfilterApp: Plot data received:', result);
            this.plotData = result;
            this.renderTemporalCDF(result);
            this.clearMessages();
            console.log('CrossfilterApp: Plot rendering complete');
        } catch (error) {
            this.showError('Failed to load plot data: ' + error.message);
        }
    }

    renderTemporalCDF(plotData) {
        console.log('CrossfilterApp: renderTemporalCDF() called with data:', plotData);
        const plotContainer = document.getElementById('plotContainer');
        
        if (!plotContainer) {
            console.error('CrossfilterApp: Plot container not found!');
            this.showError('Plot container not found');
            return;
        }
        
        try {
            console.log('CrossfilterApp: Clearing plot container and rendering...');
            // Clear any existing content (including the "No data loaded" message)
            plotContainer.innerHTML = '';
            
            // Create the plot using Plotly
            const figure = plotData.plotly_plot;
            console.log('CrossfilterApp: Plotly figure data:', figure);
            
            
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
            console.log('CrossfilterApp: Plotly.newPlot() completed successfully');

            // Handle plot selection events
            plotContainer.on('plotly_selected', (eventData) => {
                this.handlePlotSelection(eventData);
            });

            plotContainer.on('plotly_deselect', () => {
                this.clearSelection();
            });

            // Show plot controls now that plot is rendered
            const plotControls = document.getElementById('plotControls');
            if (plotControls) {
                plotControls.style.display = 'flex';
            }
            this.updateFilterButton();
        } catch (error) {
            this.showError('Failed to render plot: ' + error.message);
            console.error('Plot rendering error:', error);
        }
    }

    handlePlotSelection(eventData) {
        if (!eventData || !eventData.points) {
            return;
        }

        // Extract row indices from selected points
        // The backend should provide df_id (row index) in the customdata
        const selectedIndices = new Set();
        eventData.points.forEach((point, index) => {
            let df_id = null;
            
            // Try to get df_id from customdata first
            if (point.customdata && point.customdata.df_id !== undefined) {
                df_id = point.customdata.df_id;
            } 
            // Fallback: For individual points, pointNumber should correspond to df_id
            else if (point.pointNumber !== undefined) {
                df_id = point.pointNumber;
            }
            
            if (df_id !== null) {
                selectedIndices.add(df_id);
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
            
            // SSE will handle the refresh automatically when the filter_reset event is received
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
        const filterButton = document.getElementById('filterToSelectedBtn');
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
            
            // SSE will handle the refresh automatically when the filter_applied event is received
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


function resetFilters() {
    app.resetFilters();
}

function filterTemporalToSelected() {
    app.filterTemporalToSelected();
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    app = new CrossfilterApp();
    window.app = app;  // Also expose on window for testing
});