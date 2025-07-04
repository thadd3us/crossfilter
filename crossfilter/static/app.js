// Crossfilter frontend application

// ProjectionType enum to match backend schema.py
const ProjectionType = {
    GEO: 'geo',
    TEMPORAL: 'temporal'
};

class CrossfilterApp {
    constructor() {
        this.selectedTemporalRowIndices = new Set();
        this.selectedGeoRowIndices = new Set();
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
                    <strong>Status:</strong> ${status.row_count} rows loaded with ${status.columns.length} columns (${status.memory_usage_mb} MB), ${status.filtered_count} remain in current view
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
            // console.log('CrossfilterApp: Received SSE event:', data);
            
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
            this.showInfo('Loading temporal and geographic plots...');
            
            // Load both plots simultaneously
            const [temporalResponse, geoResponse] = await Promise.all([
                fetch('/api/plots/temporal?max_groups=10000'),
                fetch('/api/plots/geo')
            ]);
            
            if (!temporalResponse.ok) {
                throw new Error(`Temporal plot HTTP error! status: ${temporalResponse.status}`);
            }
            if (!geoResponse.ok) {
                throw new Error(`Geo plot HTTP error! status: ${geoResponse.status}`);
            }
            
            const [temporalResult, geoResult] = await Promise.all([
                temporalResponse.json(),
                geoResponse.json()
            ]);
            
            console.log('CrossfilterApp: Both plot data received');
            this.plotData = temporalResult;
            this.renderTemporalCDF(temporalResult);
            this.renderGeoPlot(geoResult);

            this.clearMessages();
            console.log('CrossfilterApp: Plot rendering complete');
        } catch (error) {
            this.showError('Failed to load plot data: ' + error.message);
        }
    }

    renderTemporalCDF(plotData) {
        console.log('CrossfilterApp: renderTemporalCDF() called with data:', plotData);
        const plotContainer = document.getElementById('plotContainer');
        const statusElement = document.getElementById('temporalPlotStatus');
        
        if (!plotContainer || !statusElement) {
            console.error('CrossfilterApp: Plot container or status element not found!');
            this.showError('Plot container not found');
            return;
        }
        
        try {
            console.log('CrossfilterApp: Clearing plot container and rendering...');
            // Clear any existing content (including the "No data loaded" message)
            plotContainer.innerHTML = '';
            
            // Update status line
            const aggregationText = plotData.aggregation_level || 'None';
            statusElement.innerHTML = `
                Showing ${plotData.point_count} buckets representing ${plotData.distinct_point_count} distinct points, aggregated at ${aggregationText}
            `;
            statusElement.style.display = 'block';
            
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
                this.handleTemporalPlotSelection(eventData);
            });

            plotContainer.on('plotly_deselect', () => {
                this.clearTemporalSelection();
            });

            // Show plot controls now that plot is rendered
            const plotControls = document.getElementById('plotControls');
            if (plotControls) {
                plotControls.style.display = 'flex';
            }
            this.updateTemporalFilterButton();
        } catch (error) {
            this.showError('Failed to render plot: ' + error.message);
            console.error('Plot rendering error:', error);
        }
    }

    renderGeoPlot(plotData) {
        console.log('CrossfilterApp: renderGeoPlot() called with data:', plotData);
        const plotContainer = document.getElementById('geoPlotContainer');
        const statusElement = document.getElementById('geoPlotStatus');
        
        if (!plotContainer || !statusElement) {
            console.error('CrossfilterApp: Geo plot container or status element not found!');
            this.showError('Geo plot container not found');
            return;
        }
        
        try {
            console.log('CrossfilterApp: Clearing geo plot container and rendering...');
            // Clear any existing content
            plotContainer.innerHTML = '';
            
            // Update status line
            const aggregationText = plotData.aggregation_level || 'None';
            statusElement.innerHTML = `
                Showing ${plotData.marker_count} markers representing ${plotData.distinct_point_count} distinct points, aggregated at ${aggregationText}
            `;
            
            // Create the plot using Plotly
            const figure = plotData.plotly_plot;
            console.log('CrossfilterApp: Plotly geo figure data:', figure);
            
            // Enable selection on the plot
            const config = {
                displayModeBar: true,
                modeBarButtonsToAdd: ['select2d', 'lasso2d'],
                modeBarButtonsToRemove: ['autoScale2d'],
                displaylogo: false
            };

            Plotly.newPlot(plotContainer, figure.data, figure.layout, config);
            console.log('CrossfilterApp: Plotly.newPlot() completed successfully for geo plot');

            // Handle plot selection events
            plotContainer.on('plotly_selected', (eventData) => {
                this.handleGeoPlotSelection(eventData);
            });

            plotContainer.on('plotly_deselect', () => {
                this.clearGeoSelection();
            });

            // Show geo plot section and controls
            const geoPlotSection = document.getElementById('geoPlotSection');
            const geoPlotControls = document.getElementById('geoPlotControls');
            if (geoPlotSection) {
                geoPlotSection.style.display = 'block';
            }
            if (geoPlotControls) {
                geoPlotControls.style.display = 'flex';
            }
            this.updateGeoFilterButton();
        } catch (error) {
            this.showError('Failed to render geo plot: ' + error.message);
            console.error('Geo plot rendering error:', error);
        }
    }

    handleTemporalPlotSelection(eventData) {
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

        this.selectedTemporalRowIndices = selectedIndices;
        this.updateTemporalFilterButton();
    }

    handleGeoPlotSelection(eventData) {
        if (!eventData || !eventData.points) {
            return;
        }

        // Extract row indices from selected points
        // The backend should provide df_id (row index) in the customdata
        const selectedIndices = new Set();
        eventData.points.forEach((point, index) => {
            let df_id = null;
            
            // Try to get df_id from customdata first
            if (point.customdata && point.customdata[0] !== undefined) {
                df_id = point.customdata[0]; // First item in customdata is df_id
            } 
            // Fallback: For individual points, pointNumber should correspond to df_id
            else if (point.pointNumber !== undefined) {
                df_id = point.pointNumber;
            }
            
            if (df_id !== null) {
                selectedIndices.add(df_id);
            }
        });

        this.selectedGeoRowIndices = selectedIndices;
        this.updateGeoFilterButton();
    }

    clearTemporalSelection() {
        this.selectedTemporalRowIndices.clear();
        this.updateTemporalFilterButton();
    }

    clearGeoSelection() {
        this.selectedGeoRowIndices.clear();
        this.updateGeoFilterButton();
    }

    clearSelection() {
        this.clearTemporalSelection();
        this.clearGeoSelection();
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

    updateTemporalFilterButton() {
        const filterButton = document.getElementById('filterToSelectedBtn');
        const plotSelectionInfo = document.getElementById('plotSelectionInfo');
        
        if (this.selectedTemporalRowIndices.size > 0) {
            filterButton.disabled = false;
            plotSelectionInfo.textContent = `Selected: ${this.selectedTemporalRowIndices.size} points`;
        } else {
            filterButton.disabled = true;
            plotSelectionInfo.textContent = '';
        }
    }

    updateGeoFilterButton() {
        const filterButton = document.getElementById('filterToSelectedGeoBtn');
        const plotSelectionInfo = document.getElementById('geoPlotSelectionInfo');
        
        if (this.selectedGeoRowIndices.size > 0) {
            filterButton.disabled = false;
            plotSelectionInfo.textContent = `Selected: ${this.selectedGeoRowIndices.size} points`;
        } else {
            filterButton.disabled = true;
            plotSelectionInfo.textContent = '';
        }
    }

    async filterToSelected(eventSource) {
        // Get the appropriate selected indices based on event source
        let selectedIndices;
        if (eventSource === ProjectionType.TEMPORAL) {
            selectedIndices = Array.from(this.selectedTemporalRowIndices);
        } else if (eventSource === ProjectionType.GEO) {
            selectedIndices = Array.from(this.selectedGeoRowIndices);
        } else {
            this.showError('Unknown event source: ' + eventSource);
            return;
        }
        
        if (selectedIndices.length === 0) {
            this.showError('No points selected. Use lasso or box select to choose points first.');
            return;
        }

        // Get display names for different event sources
        const displayNames = {
            [ProjectionType.TEMPORAL]: 'temporal buckets',
            [ProjectionType.GEO]: 'spatial regions'
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
                    event_source: eventSource
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
        return this.filterToSelected(ProjectionType.TEMPORAL);
    }

    async filterGeoToSelected() {
        return this.filterToSelected(ProjectionType.GEO);
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

function filterGeoToSelected() {
    app.filterGeoToSelected();
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    app = new CrossfilterApp();
    window.app = app;  // Also expose on window for testing
});