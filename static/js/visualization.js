class ProteinVisualizer {
    constructor(viewerId, contactMapId) {
        this.viewerId = viewerId;
        this.contactMapId = contactMapId;
        this.viewer = null;
        this.currentStructure = null;
        this.selectedResidues = new Set();
        this.confidenceThresholds = {
            low: 50,
            medium: 70,
            high: 90
        };
        this.annotations = new Map();
        this.customColorScheme = null;
        this.heatmaps = new Map();
        this.activeHeatmap = null;
        this.visibleAnnotations = {
            domains: true,
            'active-sites': true,
            'binding-sites': true
        };

        // Performance monitoring
        this.performanceMetrics = {
            lastRenderTime: 0,
            frameCount: 0,
            averageRenderTime: 0
        };

        // Initialize logging
        this.debugMode = true;
        this.logPerformance = true;

        this.initViewer();
        this.setupControls();
    }

    initViewer() {
        try {
            const element = document.getElementById(this.viewerId);
            if (!element) {
                throw new Error('Viewer container not found');
            }

            // Initialize viewer with proper configuration
            this.viewer = $3Dmol.createViewer($(element), {
                defaultcolors: $3Dmol.rasmolElementColors,
                backgroundColor: 'white',
                antialias: true,
                disableFog: true
            });

            if (!this.viewer) {
                throw new Error('Failed to create viewer');
            }

            console.log('Viewer initialized successfully');

            // Add mouse wheel zoom with proper event handling
            element.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY * -0.001;
                if (this.viewer) {
                    this.viewer.zoom(delta);
                    this.viewer.render();
                }
            });

            // Add touch controls for mobile devices
            let touchStartX, touchStartY;
            element.addEventListener('touchstart', (e) => {
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
            });

            element.addEventListener('touchmove', (e) => {
                const deltaX = (e.touches[0].clientX - touchStartX) * 0.5;
                const deltaY = (e.touches[0].clientY - touchStartY) * 0.5;
                this.viewer.rotate(deltaX, deltaY);
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
                this.viewer.render();
            });

            // Initialize viewer style
            this.viewer.setStyle({}, {cartoon: {color: 'spectrum'}});
            this.viewer.render();
        } catch (error) {
            console.error('Error initializing viewer:', error);
            throw error;
        }
    }

    setupControls() {
        try {
            // Add keyboard controls with proper viewer check
            document.addEventListener('keydown', (e) => {
                if (!this.viewer) {
                    console.error('Viewer not initialized');
                    return;
                }

                const rotationSpeed = 10;
                const zoomSpeed = 0.1;

                switch(e.key) {
                    case 'ArrowLeft':
                        this.viewer.rotate(rotationSpeed, {x: 0, y: 1, z: 0});
                        break;
                    case 'ArrowRight':
                        this.viewer.rotate(-rotationSpeed, {x: 0, y: 1, z: 0});
                        break;
                    case 'ArrowUp':
                        this.viewer.rotate(rotationSpeed, {x: 1, y: 0, z: 0});
                        break;
                    case 'ArrowDown':
                        this.viewer.rotate(-rotationSpeed, {x: 1, y: 0, z: 0});
                        break;
                    case '+':
                        this.viewer.zoom(zoomSpeed);
                        break;
                    case '-':
                        this.viewer.zoom(-zoomSpeed);
                        break;
                }
                this.viewer.render();
            });

            // Add annotation controls
            const annotationTypes = ['domains', 'active-sites', 'binding-sites'];
            annotationTypes.forEach(type => {
                const checkbox = document.getElementById(`show-${type}`);
                if (checkbox) {
                    checkbox.addEventListener('change', () => {
                        this.visibleAnnotations[type] = checkbox.checked;
                        this.updateVisualization();
                    });
                }
            });

            // Add surface toggle
            const surfaceToggle = document.getElementById('toggle-surface');
            if (surfaceToggle) {
                surfaceToggle.addEventListener('click', () => {
                    this.toggleSurface();
                });
            }

            // Add reset view button
            const resetButton = document.getElementById('reset-view');
            if (resetButton) {
                resetButton.addEventListener('click', () => {
                    this.resetView();
                });
            }

            console.log('Controls setup successfully');
        } catch (error) {
            console.error('Error setting up controls:', error);
        }
    }

    addHeatmap(name, data, annotations = null) {
        try {
            // Validate input data
            if (!data || !Array.isArray(data)) {
                console.error(`Invalid data for heatmap ${name}`);
                return false;
            }

            const defaultColorScale = [
                {value: 0, color: '#FF0000'},    // Red
                {value: 0.33, color: '#FFA500'}, // Orange
                {value: 0.67, color: '#FFFF00'}, // Yellow
                {value: 1, color: '#00FF00'}     // Green
            ];

            this.heatmaps.set(name, {
                data: data,
                annotations: annotations,
                colorScale: defaultColorScale
            });

            console.log(`Added heatmap ${name} with ${data.length} values`);
            return true;
        } catch (error) {
            console.error(`Error adding heatmap ${name}:`, error);
            return false;
        }
    }

    showHeatmap(name) {
        try {
            if (!this.heatmaps.has(name) || !this.currentStructure) {
                console.error(`Invalid heatmap ${name} or no structure loaded`);
                return;
            }

            const heatmap = this.heatmaps.get(name);
            this.activeHeatmap = name;

            // Apply coloring scheme with proper gradient handling
            const colorScheme = {
                prop: 'b',
                gradient: new $3Dmol.Gradient.RWB(0, 1),
                min: 0,
                max: 1
            };

            // Update atom properties with proper data mapping
            const atoms = this.currentStructure.atoms();
            atoms.forEach((atom, index) => {
                if (atom.atom === 'CA') { // Only color alpha carbons
                    const residueIndex = atom.resi - 1;
                    if (residueIndex >= 0 && residueIndex < heatmap.data.length) {
                        atom.b = heatmap.data[residueIndex];
                    }
                }
            });

            // Apply the color scheme to the structure
            this.currentStructure.setStyle({}, {
                cartoon: {
                    colorscheme: colorScheme
                }
            });

            // Add annotation labels if available
            if (heatmap.annotations) {
                this.addResidueLabels(heatmap.annotations);
            }

            this.viewer.render();
            console.log(`Heatmap ${name} applied successfully with ${heatmap.data.length} residues`);
        } catch (error) {
            console.error('Error showing heatmap:', error);
        }
    }

    toggleAnnotation(type, visible) {
        this.visibleAnnotations[type] = visible;
        this.updateVisualization();
    }

    addDomainAnnotation(start, end, label, color) {
        for (let i = start; i <= end; i++) {
            this.annotations.set(i, {
                type: 'domains',
                text: label,
                color: color || '#ffeb3b'
            });
        }
        this.updateVisualization();
    }

    addActiveSiteAnnotation(position, label) {
        this.annotations.set(position, {
            type: 'active-sites',
            text: `Active Site: ${label}`,
            color: '#f44336'
        });
        this.updateVisualization();
    }

    addBindingSiteAnnotation(position, label) {
        this.annotations.set(position, {
            type: 'binding-sites',
            text: `Binding Site: ${label}`,
            color: '#2196f3'
        });
        this.updateVisualization();
    }

    updateVisualization() {
        if (!this.currentStructure) return;

        // Reset style
        this.currentStructure.setStyle({}, {
            cartoon: {
                colorscheme: this.activeHeatmap ?
                    {
                        prop: 'b',
                        gradient: this.heatmaps.get(this.activeHeatmap).colorScale,
                        min: 0,
                        max: 1
                    } :
                    {color: 'spectrum'}
            }
        });

        // Apply visible annotations
        this.annotations.forEach((annotation, residue) => {
            if (this.visibleAnnotations[annotation.type]) {
                this.currentStructure.setStyle({resi: residue}, {
                    cartoon: {color: annotation.color}
                });
            }
        });

        this.viewer.render();
    }

    visualizeStructure(pdbData, confidenceData) {
        try {
            console.time('structure-visualization');
            this.log('Starting structure visualization...');

            // Clear previous structure
            this.viewer.clear();

            // Load new structure
            try {
                this.currentStructure = this.viewer.addModel(pdbData, "pdb");
                this.log('Structure loaded successfully');
            } catch (error) {
                this.logError('Failed to load structure:', error);
                return false;
            }

            // Apply confidence coloring
            if (confidenceData && Array.isArray(confidenceData)) {
                const normalizedConfidence = confidenceData.map(score =>
                    Math.min(100, Math.max(0, score))
                );

                try {
                    this.currentStructure.setStyle({}, {
                        cartoon: {
                            color: this.generateConfidenceColors(normalizedConfidence)
                        }
                    });
                    this.log('Confidence coloring applied');
                } catch (error) {
                    this.logError('Failed to apply confidence coloring:', error);
                }
            }

            // Update view
            this.viewer.zoomTo();
            this.viewer.render();

            // Log performance metrics
            if (this.logPerformance) {
                const renderTime = performance.now() - this.performanceMetrics.lastRenderTime;
                this.performanceMetrics.frameCount++;
                this.performanceMetrics.averageRenderTime =
                    (this.performanceMetrics.averageRenderTime * (this.performanceMetrics.frameCount - 1) + renderTime)
                    / this.performanceMetrics.frameCount;
                this.performanceMetrics.lastRenderTime = performance.now();
                this.log(`Render metrics - Time: ${renderTime.toFixed(2)}ms, Avg: ${this.performanceMetrics.averageRenderTime.toFixed(2)}ms`);
            }

            console.timeEnd('structure-visualization');
            return true;
        } catch (error) {
            this.logError('Critical error in structure visualization:', error);
            return false;
        }
    }

    log(message) {
        if (this.debugMode) {
            console.log(`[ProteinVisualizer] ${message}`);
        }
    }

    logError(message, error) {
        console.error(`[ProteinVisualizer] ${message}`, error);
    }

    generateConfidenceColors(score) {
        try {
            // Map confidence score to color based on thresholds
            if (score >= this.confidenceThresholds.high) {
                return '#00FF00';  // Green for high confidence
            } else if (score >= this.confidenceThresholds.medium) {
                return '#FFFF00';  // Yellow for medium confidence
            } else if (score >= this.confidenceThresholds.low) {
                return '#FFA500';  // Orange for low confidence
            } else {
                return '#FF0000';  // Red for very low confidence
            }
        } catch (error) {
            console.error('Error generating confidence colors:', error);
            return '#808080';  // Default gray color on error
        }
    }

    updateContactMap(contactMap, annotations) {
        try {
            if (!contactMap || !Array.isArray(contactMap)) {
                console.error('Invalid contact map data');
                return;
            }

            const data = [{
                z: contactMap,
                type: 'heatmap',
                colorscale: 'Viridis',
                showscale: true
            }];

            // Add annotations if provided
            if (annotations && Array.isArray(annotations)) {
                data[0].text = annotations;
                data[0].hoverongaps = false;
                data[0].hoverinfo = 'text';
            }

            const layout = {
                title: 'Contact Map',
                xaxis: {
                    title: 'Residue Index',
                    showgrid: false
                },
                yaxis: {
                    title: 'Residue Index',
                    showgrid: false
                },
                width: 500,
                height: 500
            };

            Plotly.newPlot(this.contactMapId, data, layout);
            console.log('Contact map updated successfully');
        } catch (error) {
            console.error('Error updating contact map:', error);
        }
    }

    addSurfaceAnalysis() {
        if (!this.currentStructure) return;

        try {
            // Calculate and add the molecular surface
            this.viewer.addSurface($3Dmol.SurfaceType.VDW, {
                opacity: 0.7,
                colorscheme: {
                    prop: 'b',
                    gradient: this.activeHeatmap && this.heatmaps.has(this.activeHeatmap) ?
                        this.heatmaps.get(this.activeHeatmap).colorScale :
                        this.generateConfidenceColors()
                }
            }, {model: this.currentStructure});
        } catch (error) {
            console.error("Error generating surface analysis:", error);
        }
    }

    toggleSurface() {
        if (!this.currentStructure) return;

        try {
            const surfaces = this.viewer.getSurfacesFor(this.currentStructure);
            if (surfaces && surfaces.length > 0) {
                surfaces.forEach(surface => {
                    try {
                        this.viewer.removeSurface(surface);
                    } catch (error) {
                        console.error('Error removing surface:', error);
                    }
                });
            } else {
                this.addSurfaceAnalysis();
            }
            this.viewer.render();
        } catch (error) {
            console.error('Error toggling surface:', error);
        }
    }

    resetView() {
        this.viewer.zoomTo();
        this.selectedResidues.clear();
        this.annotations.clear();
        if (this.currentStructure) {
            this.visualizeStructure(this.currentStructure.toPDB(), null);
        }
        this.viewer.render();
    }

    handleResidueClick(residue, atom) {
        try {
            if (!atom || !this.currentStructure) return;

            if (this.selectedResidues.has(residue)) {
                this.selectedResidues.delete(residue);
                this.currentStructure.setStyle({resi: residue}, {
                    cartoon: this.activeHeatmap ?
                        { colorscheme: { prop: 'b' } } :
                        { color: 'spectrum' }
                });
            } else {
                this.selectedResidues.add(residue);
                const value = this.activeHeatmap && this.heatmaps.has(this.activeHeatmap) ?
                    this.heatmaps.get(this.activeHeatmap).data[residue - 1] :
                    atom.b || 0;
                this.currentStructure.setStyle({resi: residue}, {
                    cartoon: { color: 'yellow' }
                });
                this.viewer.addLabel(`${this.activeHeatmap || 'Confidence'}: ${value.toFixed(2)}`, {
                    position: atom,
                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                    fontColor: '#000000'
                });
            }
            this.viewer.render();
        } catch (error) {
            console.error('Error handling residue click:', error);
        }
    }

    addResidueLabels() {
        if (!this.currentStructure) return;

        this.viewer.removeAllLabels();
        this.currentStructure.atoms().forEach(atom => {
            if (atom.atom === 'CA') {
                const residue = atom.resi;
                if (this.annotations.has(residue)) {
                    const annotation = this.annotations.get(residue);
                    this.viewer.addLabel(annotation.text, {
                        position: atom,
                        backgroundColor: 'rgba(255, 255, 255, 0.8)',
                        fontColor: '#000000',
                        fontSize: 12,
                        borderWidth: 1
                    });
                }
            }
        });
    }
}
