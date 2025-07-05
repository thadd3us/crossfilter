// Crossfilter frontend application

// ProjectionType enum to match backend schema.py
const ProjectionType = {
    GEO: 'geo',
    TEMPORAL: 'temporal'
};

// FilterOperatorType enum to match backend schema.py
const FilterOperatorType = {
    INTERSECTION: 'intersection',
    SUBTRACTION: 'subtraction'
};

// Class to manage plot selection data and UI elements for a specific plot
class PlotSelectionData {
    constructor(projectionType, intersectionBtnId, subtractionBtnId, selectionInfoId) {
        this.projectionType = projectionType;
        this.intersectionBtnId = intersectionBtnId;
        this.subtractionBtnId = subtractionBtnId;
        this.selectionInfoId = selectionInfoId;
        this.selectedDfIds = new Set();
        this.selectedCount = 0;
    }

    clearSelection() {
        this.selectedDfIds.clear();
        this.selectedCount = 0;
    }

    updateSelection(selectedDfIds, selectedCount) {
        this.selectedDfIds = selectedDfIds;
        this.selectedCount = selectedCount;
    }

    updateUI(totalRowCount) {
        const intersectionBtn = document.getElementById(this.intersectionBtnId);
        const subtractionBtn = document.getElementById(this.subtractionBtnId);
        const selectionInfo = document.getElementById(this.selectionInfoId);
        
        const hasSelection = this.selectedDfIds.size > 0;
        
        if (intersectionBtn) intersectionBtn.disabled = !hasSelection;
        if (subtractionBtn) subtractionBtn.disabled = !hasSelection;
        
        if (selectionInfo) {
            if (hasSelection) {
                const percent = ((this.selectedCount / totalRowCount) * 100).toFixed(1);
                selectionInfo.textContent = `Selected ${this.selectedCount} (${percent}%) of ${totalRowCount} rows`;
            } else {
                selectionInfo.textContent = '';
            }
        }
    }
}

class CrossfilterApp {
    constructor() {
        // Create plot selection data managers
        this.plotSelections = {
            [ProjectionType.TEMPORAL]: new PlotSelectionData(
                ProjectionType.TEMPORAL,
                'filterTemporalIntersectionBtn',
                'filterTemporalSubtractionBtn', 
                'temporalPlotSelectionInfo'
            ),
            [ProjectionType.GEO]: new PlotSelectionData(
                ProjectionType.GEO,
                'filterGeoIntersectionBtn',
                'filterGeoSubtractionBtn',
                'geoPlotSelectionInfo'
            )
        };
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
                let percent_remaining = (status.filtered_count / status.row_count * 100).toFixed(1);
                statusElement.innerHTML = `
                    <strong>Status:</strong> ${status.filtered_count} (${percent_remaining}%) of ${status.row_count} rows loaded (${status.columns.length} cols, ${status.memory_usage_mb} MB)
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
                Showing ${plotData.distinct_point_count} rows, aggregated at ${aggregationText}
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
                Showing ${plotData.distinct_point_count} rows, aggregated at ${aggregationText}
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

    handlePlotSelection(eventData, projectionType) {
        if (!eventData || !eventData.points) {
            return;
        }

        // Extract row indices from selected points
        // The backend provides df_id (row index) in customdata[0] and count in customdata[1]
        const selectedIndices = new Set();
        let selectedCount = 0;
        eventData.points.forEach((point, index) => {
            selectedIndices.add(point.customdata[0]);
            selectedCount += point.customdata[1];
        });

        console.log(`CrossfilterApp: handlePlotSelection(${projectionType}) selectedCount:`, selectedCount);

        // Update selection state using PlotSelectionData
        const plotSelection = this.plotSelections[projectionType];
        if (plotSelection) {
            plotSelection.updateSelection(selectedIndices, selectedCount);
            this.updateFilterButtons();
        }
    }

    handleTemporalPlotSelection(eventData) {
        this.handlePlotSelection(eventData, ProjectionType.TEMPORAL);
    }

    handleGeoPlotSelection(eventData) {
        this.handlePlotSelection(eventData, ProjectionType.GEO);
    }

    clearSelection(projectionType = null) {
        if (projectionType === null) {
            // Clear all selections
            Object.values(this.plotSelections).forEach(plotSelection => {
                plotSelection.clearSelection();
            });
        } else {
            // Clear specific projection selection
            const plotSelection = this.plotSelections[projectionType];
            if (plotSelection) {
                plotSelection.clearSelection();
            }
        }
        this.updateFilterButtons();
    }

    clearTemporalSelection() {
        this.clearSelection(ProjectionType.TEMPORAL);
    }

    clearGeoSelection() {
        this.clearSelection(ProjectionType.GEO);
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

    updateFilterButtons() {
        const totalRowCount = this.getFilteredRowCount();
        Object.values(this.plotSelections).forEach(plotSelection => {
            plotSelection.updateUI(totalRowCount);
        });
    }

    getFilteredRowCount() {
        // Get the current filtered row count from status
        const statusElement = document.getElementById('status');
        if (!statusElement) return 0;
        
        const statusText = statusElement.textContent;
        // Match pattern: "Status: 100 (100.0%) of 100 rows loaded"
        const match = statusText.match(/Status:\s*(\d+)\s*\(/);
        return match ? parseInt(match[1]) : 0;
    }

    // Legacy methods for backward compatibility (now simplified)
    updateTemporalFilterButton() {
        this.updateFilterButtons();
    }

    updateGeoFilterButton() {
        this.updateFilterButtons();
    }

    async filterToSelected(eventSource, filterOperator) {
        // Get the appropriate selected indices based on event source
        const plotSelection = this.plotSelections[eventSource];
        if (!plotSelection) {
            this.showError('Unknown event source: ' + eventSource);
            return;
        }
        
        const selectedIndices = Array.from(plotSelection.selectedDfIds);
        
        if (selectedIndices.length === 0) {
            this.showError('No markers selected. Use lasso or box select to choose markers first.');
            return;
        }

        // Get display names for different operations
        const operationNames = {
            [FilterOperatorType.INTERSECTION]: 'keeping only',
            [FilterOperatorType.SUBTRACTION]: 'removing'
        };
        
        const operationName = operationNames[filterOperator] || 'filtering';

        try {
            this.showInfo(`${operationName} ${selectedIndices.length} selected points...`);
            
            const response = await fetch('/api/filters/df_ids', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    df_ids: selectedIndices,
                    event_source: eventSource,
                    filter_operator: filterOperator
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showInfo(`Filter applied: ${result.filter_state.filtered_count} rows remaining`);
            
            // SSE will handle the refresh automatically when the filter_applied event is received
        } catch (error) {
            this.showError(`Failed to apply ${eventSource} ${filterOperator} filter: ` + error.message);
        }
    }

    // Methods for intersection operations
    async filterTemporalIntersection() {
        return this.filterToSelected(ProjectionType.TEMPORAL, FilterOperatorType.INTERSECTION);
    }

    async filterGeoIntersection() {
        return this.filterToSelected(ProjectionType.GEO, FilterOperatorType.INTERSECTION);
    }

    // Methods for subtraction operations
    async filterTemporalSubtraction() {
        return this.filterToSelected(ProjectionType.TEMPORAL, FilterOperatorType.SUBTRACTION);
    }

    async filterGeoSubtraction() {
        return this.filterToSelected(ProjectionType.GEO, FilterOperatorType.SUBTRACTION);
    }

    // Legacy methods for backward compatibility
    async filterTemporalToSelected() {
        return this.filterTemporalIntersection();
    }

    async filterGeoToSelected() {
        return this.filterGeoIntersection();
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

// Intersection filter functions
function filterTemporalIntersection() {
    app.filterTemporalIntersection();
}

function filterGeoIntersection() {
    app.filterGeoIntersection();
}

// Subtraction filter functions
function filterTemporalSubtraction() {
    app.filterTemporalSubtraction();
}

function filterGeoSubtraction() {
    app.filterGeoSubtraction();
}

// Legacy functions for backward compatibility
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