/**
 * Crossfilter Web Application
 * Interactive geographic and temporal data visualization with cross-filtering
 */

class CrossfilterApp {
    constructor() {
        this.maxGroups = 100000;
        this.keplerMap = null;
        this.temporalPlot = null;
        this.currentSpatialData = null;
        this.currentTemporalData = null;
        
        this.initializeEventListeners();
        this.initializeApp();
    }
    
    initializeEventListeners() {
        // Control buttons
        document.getElementById('refresh-plots').addEventListener('click', () => this.refreshPlots());
        document.getElementById('reset-filters').addEventListener('click', () => this.resetFilters());
        document.getElementById('undo-filter').addEventListener('click', () => this.undoFilter());
        
        // Max groups input
        document.getElementById('max-groups').addEventListener('change', (e) => {
            this.maxGroups = parseInt(e.target.value);
            this.refreshPlots();
        });
        
        // Plot-specific controls
        document.getElementById('geo-filter-visible').addEventListener('click', () => this.filterGeoToVisible());
        document.getElementById('geo-clear-selection').addEventListener('click', () => this.clearGeoSelection());
        document.getElementById('temporal-filter-visible').addEventListener('click', () => this.filterTemporalToVisible());
        document.getElementById('temporal-clear-selection').addEventListener('click', () => this.clearTemporalSelection());
    }
    
    async initializeApp() {
        this.updateStatus('Initializing application...');
        
        try {
            // Check session status
            const sessionStatus = await this.fetchAPI('/api/session');
            
            if (sessionStatus.status === 'loaded') {
                this.updateStatus(`Loaded ${sessionStatus.shape[0]} records - ${sessionStatus.filter_state.filtered_count} visible`);
                await this.refreshPlots();
            } else {
                this.updateStatus('No data loaded. Please start the server with --preload_jsonl option.');
            }
        } catch (error) {
            this.showError('Failed to initialize application', error);
        }
    }
    
    async refreshPlots() {
        this.updateStatus('Refreshing visualizations...');
        
        try {
            // Load both plots in parallel
            const [spatialResponse, temporalResponse] = await Promise.all([
                this.fetchAPI(`/api/plots/spatial?max_groups=${this.maxGroups}`),
                this.fetchAPI(`/api/plots/temporal?max_groups=${this.maxGroups}`)
            ]);
            
            // Update geographic visualization
            await this.updateGeographicPlot(spatialResponse);
            
            // Update temporal visualization
            await this.updateTemporalPlot(temporalResponse);
            
            // Update status
            const sessionStatus = await this.fetchAPI('/api/session');
            this.updateStatus(
                `${sessionStatus.filter_state.filtered_count} of ${sessionStatus.filter_state.total_count} points visible | ` +
                `Geo: ${spatialResponse.data_type} (${spatialResponse.point_count} groups) | ` +
                `Temporal: ${temporalResponse.data_type} (${temporalResponse.point_count} groups)`
            );
            
        } catch (error) {
            this.showError('Failed to refresh plots', error);
        }
    }
    
    async updateGeographicPlot(spatialResponse) {
        const container = document.getElementById('geographic-plot');
        
        try {
            // Try to use Kepler.gl first
            if (window.KeplerGl && spatialResponse.kepler_data && spatialResponse.kepler_config) {
                await this.initializeKeplerMap(container, spatialResponse);
            } else {
                // Fallback to Plotly
                this.initializePlotlyGeo(container, spatialResponse.plotly_fallback);
            }
            
            this.currentSpatialData = spatialResponse;
            
        } catch (error) {
            console.warn('Kepler.gl failed, falling back to Plotly:', error);
            this.initializePlotlyGeo(container, spatialResponse.plotly_fallback);
            this.currentSpatialData = spatialResponse;
        }
    }
    
    async initializeKeplerMap(container, spatialResponse) {
        // Clear container
        container.innerHTML = '<div class="kepler-container"></div>';
        const keplerContainer = container.querySelector('.kepler-container');
        
        // Initialize Kepler.gl
        this.keplerMap = new window.KeplerGl({
            container: keplerContainer,
            mapboxApiAccessToken: '', // Add your Mapbox token if needed
            width: container.offsetWidth,
            height: container.offsetHeight
        });
        
        // Add data and apply config
        this.keplerMap.addDataToMap({
            datasets: [{
                info: {id: 'crossfilter_data', label: 'Crossfilter Data'},
                data: {
                    fields: this.extractKeplerFields(spatialResponse.kepler_data),
                    rows: this.extractKeplerRows(spatialResponse.kepler_data)
                }
            }],
            config: spatialResponse.kepler_config.config || {}
        });
        
        // Set up event listeners for selection
        this.setupKeplerEventListeners();
    }
    
    initializePlotlyGeo(container, plotlyConfig) {
        container.innerHTML = '<div id="plotly-geo-plot" style="width: 100%; height: 100%;"></div>';
        const plotDiv = container.querySelector('#plotly-geo-plot');
        
        Plotly.newPlot(plotDiv, plotlyConfig.data, plotlyConfig.layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d']
        });
        
        // Set up selection event listener
        plotDiv.on('plotly_selected', (eventData) => {
            if (eventData && eventData.points) {
                const selectedUuids = eventData.points.map(point => point.customdata || point.pointIndex);
                this.handleGeoSelection(selectedUuids);
            }
        });
    }
    
    async updateTemporalPlot(temporalResponse) {
        const container = document.getElementById('temporal-plot');
        container.innerHTML = '<div id="plotly-temporal-plot" style="width: 100%; height: 100%;"></div>';
        const plotDiv = container.querySelector('#plotly-temporal-plot');
        
        Plotly.newPlot(plotDiv, temporalResponse.plotly_plot.data, temporalResponse.plotly_plot.layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d']
        });
        
        // Set up selection event listener
        plotDiv.on('plotly_selected', (eventData) => {
            if (eventData && eventData.points) {
                this.handleTemporalSelection(eventData.points);
            }
        });
        
        // Set up zoom/pan event listeners for "filter to visible"
        plotDiv.on('plotly_relayout', (eventData) => {
            this.handleTemporalViewChange(eventData);
        });
        
        this.currentTemporalData = temporalResponse;
        this.temporalPlot = plotDiv;
    }
    
    extractKeplerFields(data) {
        if (!data || data.length === 0) return [];
        
        const sample = data[0];
        return Object.keys(sample).map(key => ({
            name: key,
            type: this.inferKeplerFieldType(sample[key])
        }));
    }
    
    extractKeplerRows(data) {
        if (!data || data.length === 0) return [];
        
        const fields = this.extractKeplerFields(data);
        return data.map(row => fields.map(field => row[field.name]));
    }
    
    inferKeplerFieldType(value) {
        if (typeof value === 'number') return 'integer';
        if (typeof value === 'string') return 'string';
        if (value instanceof Date) return 'timestamp';
        return 'string';
    }
    
    setupKeplerEventListeners() {
        // This would require more sophisticated integration with Kepler.gl's state management
        // For now, we'll rely on the plot-level controls
    }
    
    async handleGeoSelection(selectedUuids) {
        if (!selectedUuids || selectedUuids.length === 0) return;
        
        try {
            await this.applyFilter(selectedUuids, 'spatial', 'Geographic selection');
            await this.refreshPlots();
        } catch (error) {
            this.showError('Failed to apply geographic filter', error);
        }
    }
    
    async handleTemporalSelection(selectedPoints) {
        if (!selectedPoints || selectedPoints.length === 0) return;
        
        // Extract UUIDs from the temporal selection
        const selectedUuids = [];
        selectedPoints.forEach(point => {
            if (this.currentTemporalData.data_type === 'individual') {
                selectedUuids.push(point.customdata);
            } else {
                // For aggregated data, we need to get UUIDs from the aggregation
                // This requires additional data from the server
                console.log('Temporal aggregated selection not yet implemented');
            }
        });
        
        if (selectedUuids.length > 0) {
            try {
                await this.applyFilter(selectedUuids, 'temporal', 'Temporal selection');
                await this.refreshPlots();
            } catch (error) {
                this.showError('Failed to apply temporal filter', error);
            }
        }
    }
    
    handleTemporalViewChange(eventData) {
        // Store the current view bounds for "filter to visible" functionality
        if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
            this.currentTemporalViewBounds = {
                start: eventData['xaxis.range[0]'],
                end: eventData['xaxis.range[1]']
            };
        }
    }
    
    async filterGeoToVisible() {
        // This would require integration with the map's current viewport
        // For now, show a message
        this.updateStatus('Geographic "filter to visible" not yet implemented - use selection tools');
    }
    
    async filterTemporalToVisible() {
        if (!this.currentTemporalViewBounds) {
            this.updateStatus('No temporal view bounds available - zoom/pan the plot first');
            return;
        }
        
        try {
            // This would require a server endpoint to filter by time range
            this.updateStatus('Temporal "filter to visible" not yet fully implemented');
        } catch (error) {
            this.showError('Failed to filter temporal to visible', error);
        }
    }
    
    clearGeoSelection() {
        // Clear any geographic selections
        if (this.keplerMap) {
            // Kepler.gl selection clearing would go here
        }
        
        const plotlyGeo = document.querySelector('#plotly-geo-plot');
        if (plotlyGeo) {
            Plotly.restyle(plotlyGeo, {'selectedpoints': [null]});
        }
    }
    
    clearTemporalSelection() {
        if (this.temporalPlot) {
            Plotly.restyle(this.temporalPlot, {'selectedpoints': [null]});
        }
    }
    
    async applyFilter(uuids, operationType, description) {
        const response = await this.fetchAPI('/api/filters/apply', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                uuids: uuids,
                operation_type: operationType,
                description: description
            })
        });
        
        return response;
    }
    
    async resetFilters() {
        try {
            this.updateStatus('Resetting filters...');
            await this.fetchAPI('/api/filters/reset', {method: 'POST'});
            await this.refreshPlots();
        } catch (error) {
            this.showError('Failed to reset filters', error);
        }
    }
    
    async undoFilter() {
        try {
            const response = await this.fetchAPI('/api/filters/undo', {method: 'POST'});
            if (response.success) {
                this.updateStatus('Filter undone');
                await this.refreshPlots();
            } else {
                this.updateStatus(response.message || 'No operations to undo');
            }
        } catch (error) {
            this.showError('Failed to undo filter', error);
        }
    }
    
    async fetchAPI(endpoint, options = {}) {
        const response = await fetch(endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    updateStatus(message) {
        document.getElementById('status-text').textContent = message;
    }
    
    showError(message, error) {
        console.error(message, error);
        const errorHtml = `
            <div class="error">
                <strong>${message}</strong><br>
                ${error.message || error}
            </div>
        `;
        
        // Show error in both plot containers if they're empty
        const geoContainer = document.getElementById('geographic-plot');
        const temporalContainer = document.getElementById('temporal-plot');
        
        if (!geoContainer.querySelector('.kepler-container') && !geoContainer.querySelector('#plotly-geo-plot')) {
            geoContainer.innerHTML = errorHtml;
        }
        
        if (!temporalContainer.querySelector('#plotly-temporal-plot')) {
            temporalContainer.innerHTML = errorHtml;
        }
        
        this.updateStatus(`Error: ${message}`);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new CrossfilterApp();
});