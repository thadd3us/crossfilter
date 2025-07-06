// Crossfilter Vue.js frontend application

const { createApp, ref, reactive, computed, watch, onMounted, onUnmounted } = Vue;

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

// Base class for projection state management
class ProjectionState {
    constructor(projectionType, title) {
        this.projectionType = projectionType;
        this.title = title;
        this.selectedDfIds = new Set();
        this.selectedCount = 0;
        this.totalCount = 0;
        this.isCollapsed = false;
        this.plotData = null;
        this.aggregationLevel = null;
        this.distinctPointCount = 0;
        this.plotContainer = null;
    }

    clearSelection() {
        this.selectedDfIds.clear();
        this.selectedCount = 0;
    }

    updateSelection(selectedDfIds, selectedCount) {
        this.selectedDfIds = selectedDfIds;
        this.selectedCount = selectedCount;
    }

    updatePlotData(plotData) {
        this.plotData = plotData;
        this.aggregationLevel = plotData.aggregation_level || 'None';
        this.distinctPointCount = plotData.distinct_point_count || 0;
    }

    getStatusText() {
        const parts = [];
        
        // Use distinctPointCount from plot data if available, otherwise use totalCount
        const rowCount = this.distinctPointCount || this.totalCount || 0;
        parts.push(`${rowCount} rows`);
        
        if (this.selectedCount > 0) {
            const percent = rowCount > 0 ? ((this.selectedCount / rowCount) * 100).toFixed(1) : '0.0';
            parts.push(`${this.selectedCount} (${percent}%) selected`);
        }
        
        if (this.aggregationLevel && this.aggregationLevel !== 'None') {
            parts.push(`bucket resolution: ${this.aggregationLevel}`);
        }
        
        return parts.join(', ');
    }

    hasSelection() {
        return this.selectedDfIds.size > 0;
    }

    toggleCollapse() {
        this.isCollapsed = !this.isCollapsed;
    }
}

// Specific projection state implementations
class TemporalProjectionState extends ProjectionState {
    constructor() {
        super(ProjectionType.TEMPORAL, 'Temporal CDF');
    }
}

class GeoProjectionState extends ProjectionState {
    constructor() {
        super(ProjectionType.GEO, 'Geographic Distribution');
    }
}

// Main application state
class AppState {
    constructor() {
        this.hasData = false;
        this.sessionStatus = {
            has_data: false,
            row_count: 0,
            filtered_count: 0,
            columns: [],
            memory_usage_mb: 0
        };
        this.projections = {
            [ProjectionType.TEMPORAL]: new TemporalProjectionState(),
            [ProjectionType.GEO]: new GeoProjectionState()
        };
        this.leftMenuOpen = false;
        this.eventSource = null;
        this.filterVersion = 0;
        this.messages = [];
    }

    updateSessionStatus(status) {
        this.sessionStatus = { ...status };
        this.hasData = status.has_data;
        
        // Update total count for projections
        Object.values(this.projections).forEach(projection => {
            projection.totalCount = status.row_count;
            // If we don't have plot data loaded yet, use the current filtered count
            if (!projection.distinctPointCount) {
                projection.distinctPointCount = status.filtered_count;
            }
        });
    }

    getGlobalStatusText() {
        if (!this.hasData) {
            return 'No data loaded';
        }
        
        const percentRemaining = this.sessionStatus.row_count > 0 
            ? (this.sessionStatus.filtered_count / this.sessionStatus.row_count * 100).toFixed(1)
            : '0.0';
        
        return `${this.sessionStatus.filtered_count} (${percentRemaining}%) of ${this.sessionStatus.row_count} rows loaded (${this.sessionStatus.columns.length} cols, ${this.sessionStatus.memory_usage_mb} MB)`;
    }

    showMessage(message, type = 'info', customId = null) {
        const messageObj = {
            id: customId || Date.now(),
            text: message,
            type: type
        };
        this.messages.push(messageObj);
        
        // Auto-remove info messages after 3 seconds (but not loading messages)
        if (type === 'info' && !customId) {
            setTimeout(() => {
                this.removeMessage(messageObj.id);
            }, 3000);
        }
    }

    removeMessage(id) {
        const index = this.messages.findIndex(msg => msg.id === id);
        if (index !== -1) {
            this.messages.splice(index, 1);
        }
    }

    clearMessages() {
        this.messages = [];
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showInfo(message) {
        this.showMessage(message, 'info');
    }

    toggleLeftMenu() {
        this.leftMenuOpen = !this.leftMenuOpen;
    }

    closeLeftMenu() {
        this.leftMenuOpen = false;
    }
}

// Vue components
const ProjectionComponent = {
    props: {
        projection: {
            type: Object,
            required: true
        },
        app: {
            type: Object,
            required: true
        }
    },
    template: `
        <div class="projection">
            <div 
                class="projection-header"
                :class="{ collapsed: projection.isCollapsed }"
                @click="toggleCollapse"
            >
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="projection-title">{{ projection.title }}</div>
                    <div class="projection-status">
                        Status: {{ projection.getStatusText() }}
                    </div>
                </div>
                <div class="collapse-icon" :class="{ collapsed: projection.isCollapsed }">
                    ▼
                </div>
            </div>
            
            <div 
                class="projection-content"
                v-show="!projection.isCollapsed"
            >
                <div class="projection-toolbar">
                    <button 
                        class="filter-button intersection"
                        :disabled="!projection.hasSelection()"
                        @click="filterIntersection"
                        title="Keep only selected points"
                    >
                        ∩
                    </button>
                    <button 
                        class="filter-button subtraction"
                        :disabled="!projection.hasSelection()"
                        @click="filterSubtraction"
                        title="Remove selected points"
                    >
                        −
                    </button>
                </div>
                
                <div class="plot-container" :ref="'plot-' + projection.projectionType"
                     :style="{ height: projection.projectionType === 'temporal' ? '400px' : '800px' }">
                    <div class="plot-placeholder" v-if="!projection.plotData">
                        No data loaded. Click "Load Sample Data" to begin.
                    </div>
                </div>
            </div>
        </div>
    `,
    methods: {
        toggleCollapse() {
            this.projection.toggleCollapse();
        },
        async filterIntersection() {
            await this.app.filterToSelected(this.projection.projectionType, FilterOperatorType.INTERSECTION);
        },
        async filterSubtraction() {
            await this.app.filterToSelected(this.projection.projectionType, FilterOperatorType.SUBTRACTION);
        }
    },
    mounted() {
        // Store reference to plot container
        this.projection.plotContainer = this.$refs['plot-' + this.projection.projectionType];
    }
};

const CrossfilterApp = {
    components: {
        ProjectionComponent
    },
    setup() {
        const appState = reactive(new AppState());
        
        // API methods
        const checkSessionStatus = async () => {
            try {
                console.log('CrossfilterApp: Checking session status...');
                const response = await fetch('/api/session');
                const status = await response.json();
                console.log('CrossfilterApp: Session status:', status);
                
                appState.updateSessionStatus(status);
                
                // Auto-load plot if data is already present
                if (appState.hasData) {
                    console.log('CrossfilterApp: Data found, loading plot...');
                    await refreshPlot();
                } else {
                    console.log('CrossfilterApp: No data found');
                }
            } catch (error) {
                appState.showError('Failed to check session status: ' + error.message);
            }
        };

        const setupSSEConnection = () => {
            console.log('CrossfilterApp: Setting up SSE connection...');
            
            // Close existing connection if any
            if (appState.eventSource) {
                appState.eventSource.close();
            }
            
            // Create new SSE connection
            appState.eventSource = new EventSource('/api/events/filter-changes');
            
            appState.eventSource.onopen = () => {
                console.log('CrossfilterApp: SSE connection opened');
            };
            
            appState.eventSource.onmessage = (event) => {
                handleSSEEvent(event);
            };
            
            appState.eventSource.onerror = (error) => {
                console.error('CrossfilterApp: SSE connection error:', error);
                appState.showError('Lost connection to server. Refresh page to reconnect.');
            };
        };

        const handleSSEEvent = (event) => {
            try {
                const data = JSON.parse(event.data);
                appState.filterVersion = data.version;
                
                switch (data.type) {
                    case 'connection_established':
                        console.log('CrossfilterApp: SSE connection established');
                        if (data.session_state) {
                            appState.updateSessionStatus(data.session_state);
                        }
                        break;
                        
                    case 'data_loaded':
                        console.log('CrossfilterApp: Data loaded event received');
                        appState.updateSessionStatus(data.session_state);
                        if (appState.hasData) {
                            refreshPlot();
                        }
                        break;
                        
                    case 'filter_applied':
                        console.log('CrossfilterApp: Filter applied event received');
                        appState.updateSessionStatus(data.session_state);
                        refreshPlot();
                        break;
                        
                    case 'filter_reset':
                        console.log('CrossfilterApp: Filter reset event received');
                        clearAllSelections();
                        appState.updateSessionStatus(data.session_state);
                        refreshPlot();
                        break;
                        
                    case 'heartbeat':
                        console.debug('CrossfilterApp: SSE heartbeat');
                        break;
                        
                    default:
                        console.warn('CrossfilterApp: Unknown SSE event type:', data.type);
                }
            } catch (error) {
                console.error('CrossfilterApp: Error parsing SSE event:', error);
            }
        };

        const loadSampleData = async () => {
            try {
                appState.showInfo('Loading sample data...');
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
                appState.updateSessionStatus(result.session_state);
                appState.showInfo('Sample data loaded successfully. Loading plots...');
                await refreshPlot();
            } catch (error) {
                appState.showError('Failed to load sample data: ' + error.message);
            }
        };

        const refreshPlot = async () => {
            console.log('CrossfilterApp: refreshPlot() called, hasData:', appState.hasData);
            if (!appState.hasData) {
                appState.showError('No data loaded');
                return;
            }

            try {
                console.log('CrossfilterApp: Fetching plot data...');
                const loadingMessageId = Date.now();
                appState.showMessage('Loading temporal and geographic plots...', 'info', loadingMessageId);
                
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
                
                // Update projection states
                appState.projections[ProjectionType.TEMPORAL].updatePlotData(temporalResult);
                appState.projections[ProjectionType.GEO].updatePlotData(geoResult);
                
                // Render plots
                renderPlot(appState.projections[ProjectionType.TEMPORAL], temporalResult);
                renderPlot(appState.projections[ProjectionType.GEO], geoResult);

                appState.removeMessage(loadingMessageId);
                console.log('CrossfilterApp: Plot rendering complete');
            } catch (error) {
                appState.showError('Failed to load plot data: ' + error.message);
            }
        };

        const renderPlot = (projection, plotData) => {
            console.log(`CrossfilterApp: renderPlot() called for ${projection.projectionType}`);
            
            if (!projection.plotContainer) {
                console.error('CrossfilterApp: Plot container not found!');
                return;
            }
            
            try {
                console.log('CrossfilterApp: Clearing plot container and rendering...');
                // Clear any existing content
                projection.plotContainer.innerHTML = '';
                
                // Create the plot using Plotly
                const figure = plotData.plotly_plot;
                console.log('CrossfilterApp: Plotly figure data:', figure);
                
                // Set up layout and config
                const layout = {
                    ...figure.layout,
                    title: projection.projectionType === ProjectionType.TEMPORAL 
                        ? 'Temporal Distribution (CDF)' 
                        : 'Geographic Distribution',
                    hovermode: 'closest',
                    selectdirection: projection.projectionType === ProjectionType.TEMPORAL ? 'horizontal' : undefined
                };

                const config = {
                    displayModeBar: true,
                    modeBarButtonsToAdd: ['select2d', 'lasso2d'],
                    modeBarButtonsToRemove: ['autoScale2d'],
                    displaylogo: false,
                    responsive: true
                };

                Plotly.newPlot(projection.plotContainer, figure.data, layout, config);
                console.log(`CrossfilterApp: Plotly.newPlot() completed successfully for ${projection.projectionType}`);

                // Handle plot selection events
                projection.plotContainer.on('plotly_selected', (eventData) => {
                    handlePlotSelection(eventData, projection);
                });

                projection.plotContainer.on('plotly_deselect', () => {
                    clearSelection(projection);
                });

            } catch (error) {
                appState.showError(`Failed to render ${projection.projectionType} plot: ` + error.message);
                console.error('Plot rendering error:', error);
            }
        };

        const handlePlotSelection = (eventData, projection) => {
            if (!eventData || !eventData.points) {
                return;
            }

            // Extract row indices from selected points
            const selectedIndices = new Set();
            let selectedCount = 0;
            eventData.points.forEach((point, index) => {
                selectedIndices.add(point.customdata[0]);
                selectedCount += point.customdata[1];
            });

            console.log(`CrossfilterApp: handlePlotSelection(${projection.projectionType}) selectedCount:`, selectedCount);
            projection.updateSelection(selectedIndices, selectedCount);
        };

        const clearSelection = (projection) => {
            projection.clearSelection();
        };

        const clearAllSelections = () => {
            Object.values(appState.projections).forEach(projection => {
                projection.clearSelection();
            });
        };

        const resetFilters = async () => {
            try {
                const response = await fetch('/api/filters/reset', {
                    method: 'POST'
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                clearAllSelections();
                appState.showInfo('Filters reset successfully');
                
                // SSE will handle the refresh automatically
            } catch (error) {
                appState.showError('Failed to reset filters: ' + error.message);
            }
        };

        const filterToSelected = async (eventSource, filterOperator) => {
            const projection = appState.projections[eventSource];
            if (!projection) {
                appState.showError('Unknown event source: ' + eventSource);
                return;
            }
            
            const selectedIndices = Array.from(projection.selectedDfIds);
            
            if (selectedIndices.length === 0) {
                appState.showError('No markers selected. Use lasso or box select to choose markers first.');
                return;
            }

            const operationNames = {
                [FilterOperatorType.INTERSECTION]: 'keeping only',
                [FilterOperatorType.SUBTRACTION]: 'removing'
            };
            
            const operationName = operationNames[filterOperator] || 'filtering';

            try {
                appState.showInfo(`${operationName} ${selectedIndices.length} selected points...`);
                
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
                appState.showInfo(`Filter applied: ${result.filter_state.filtered_count} rows remaining`);
                
                // SSE will handle the refresh automatically
            } catch (error) {
                appState.showError(`Failed to apply ${eventSource} ${filterOperator} filter: ` + error.message);
            }
        };

        const showAbout = () => {
            alert('Crossfilter - Interactive data exploration, filtering, and selection');
        };

        // Initialize on mount
        onMounted(async () => {
            console.log('CrossfilterApp: Initializing...');
            await checkSessionStatus();
            setupSSEConnection();
            console.log('CrossfilterApp: Initialization complete');
        });

        // Cleanup on unmount
        onUnmounted(() => {
            if (appState.eventSource) {
                appState.eventSource.close();
            }
        });

        return {
            appState,
            loadSampleData,
            resetFilters,
            filterToSelected,
            showAbout
        };
    },
    template: `
        <div class="app">
            <!-- Menu Overlay -->
            <div 
                class="menu-overlay"
                :class="{ open: appState.leftMenuOpen }"
                @click="appState.closeLeftMenu()"
            ></div>

            <!-- Left Side Menu -->
            <div class="left-menu" :class="{ open: appState.leftMenuOpen }">
                <div class="left-menu-header">
                    <h3>Menu</h3>
                    <button class="close-menu" @click="appState.closeLeftMenu()">×</button>
                </div>
                <div class="left-menu-content">
                    <div class="menu-item" @click="showAbout">
                        About Crossfilter
                    </div>
                </div>
            </div>

            <!-- Top Status Bar -->
            <div class="top-status-bar">
                <button class="hamburger-menu" @click="appState.toggleLeftMenu()">☰</button>
                <div class="status-info">
                    <strong>Status:</strong> {{ appState.getGlobalStatusText() }}
                </div>
                <div class="controls">
                    <button @click="loadSampleData">Load Sample Data</button>
                    <button @click="resetFilters" :disabled="!appState.hasData">Reset Filters</button>
                </div>
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <!-- Left Panel - Projections -->
                <div class="left-panel">
                    <ProjectionComponent 
                        v-for="projection in appState.projections" 
                        :key="projection.projectionType"
                        :projection="projection"
                        :app="{ filterToSelected }"
                    />
                </div>

                <!-- Right Panel - Content Preview -->
                <div class="right-panel">
                    <h3>Content Preview</h3>
                    <p>Preview area for selected content</p>
                </div>
            </div>

            <!-- Messages -->
            <div v-if="appState.messages.length > 0" style="position: fixed; top: 70px; right: 20px; z-index: 1000;">
                <div 
                    v-for="message in appState.messages" 
                    :key="message.id"
                    :class="['message', message.type]"
                    style="margin-bottom: 10px;"
                >
                    {{ message.text }}
                    <button @click="appState.removeMessage(message.id)" style="margin-left: 10px;">×</button>
                </div>
            </div>
        </div>
    `
};

// Create and mount the Vue app
const app = createApp(CrossfilterApp);
app.mount('#app');

// Export for backwards compatibility and testing
window.app = app;