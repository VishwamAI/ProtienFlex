class DrugDiscoveryInterface {
    constructor(proteinVisualizer) {
        this.proteinVisualizer = proteinVisualizer;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        $('#analyze-binding-sites').click(() => this.analyzeBindingSites());
        $('#predict-interactions').click(() => this.predictInteractions());
        $('#screen-off-targets').click(() => this.screenOffTargets());
    }

    analyzeBindingSites() {
        const sequence = $('#sequence-input').val();
        if (!sequence) {
            alert('Please enter a protein sequence first');
            return;
        }

        $('.drug-discovery-section .loading-spinner').addClass('active');
        $('#binding-sites-results').empty();

        $.ajax({
            url: '/analyze_binding_sites',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ sequence: sequence }),
            success: (response) => {
                $('.drug-discovery-section .loading-spinner').removeClass('active');
                this.displayBindingSites(response.binding_sites);
            },
            error: (error) => {
                $('.drug-discovery-section .loading-spinner').removeClass('active');
                alert('Error analyzing binding sites. Please try again.');
                console.error('Error:', error);
            }
        });
    }

    predictInteractions() {
        const sequence = $('#sequence-input').val();
        const ligandSmiles = $('#ligand-input').val();

        if (!sequence || !ligandSmiles) {
            alert('Please enter both a protein sequence and ligand SMILES');
            return;
        }

        $('.drug-discovery-section .loading-spinner').addClass('active');
        $('#interaction-results').empty();

        $.ajax({
            url: '/predict_interactions',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                sequence: sequence,
                ligand_smiles: ligandSmiles
            }),
            success: (response) => {
                $('.drug-discovery-section .loading-spinner').removeClass('active');
                this.displayInteractions(response);
            },
            error: (error) => {
                $('.drug-discovery-section .loading-spinner').removeClass('active');
                alert('Error predicting interactions. Please try again.');
                console.error('Error:', error);
            }
        });
    }

    screenOffTargets() {
        const sequence = $('#sequence-input').val();
        const ligandSmiles = $('#ligand-input').val();

        if (!sequence || !ligandSmiles) {
            alert('Please enter both a protein sequence and ligand SMILES');
            return;
        }

        $('.drug-discovery-section .loading-spinner').addClass('active');
        $('#off-target-results').empty();

        $.ajax({
            url: '/screen_off_targets',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                sequence: sequence,
                ligand_smiles: ligandSmiles
            }),
            success: (response) => {
                $('.drug-discovery-section .loading-spinner').removeClass('active');
                this.displayOffTargets(response.off_targets);
            },
            error: (error) => {
                $('.drug-discovery-section .loading-spinner').removeClass('active');
                alert('Error screening off-targets. Please try again.');
                console.error('Error:', error);
            }
        });
    }

    displayBindingSites(bindingSites) {
        const resultsDiv = $('#binding-sites-results');
        resultsDiv.empty();

        bindingSites.forEach((site, index) => {
            const confidenceClass = site.confidence > 0.7 ? 'high-confidence' :
                                  site.confidence > 0.3 ? 'medium-confidence' : 'low-confidence';

            const siteElement = $(`
                <div class="binding-site ${confidenceClass}">
                    <h4>Binding Site ${index + 1}</h4>
                    <p>Location: ${site.start}-${site.end}</p>
                    <p>Confidence: ${(site.confidence * 100).toFixed(1)}%</p>
                    <p>Properties:</p>
                    <ul>
                        <li>Hydrophobicity: ${site.hydrophobicity.toFixed(2)}</li>
                        <li>Surface Area: ${site.surface_area.toFixed(2)} Å²</li>
                        <li>Pocket Volume: ${site.volume.toFixed(2)} Å³</li>
                    </ul>
                </div>
            `);

            siteElement.click(() => {
                this.proteinVisualizer.highlightRegion(site.start, site.end, site.confidence);
            });

            resultsDiv.append(siteElement);
        });
    }

    displayInteractions(interactions) {
        const resultsDiv = $('#interaction-results');
        resultsDiv.empty();

        const strengthClass = interactions.binding_affinity > 0.7 ? 'high-strength' :
                            interactions.binding_affinity > 0.3 ? 'medium-strength' : 'low-strength';

        resultsDiv.append(`
            <div class="interaction-result ${strengthClass}">
                <h4>Predicted Interactions</h4>
                <p>Binding Affinity: ${(interactions.binding_affinity * 100).toFixed(1)}%</p>
                <p>Stability Score: ${(interactions.stability_score * 100).toFixed(1)}%</p>
                <h5>Key Interactions:</h5>
                <ul>
                    ${interactions.key_interactions.map(interaction => `
                        <li>${interaction.type}: ${interaction.residues.join(', ')}</li>
                    `).join('')}
                </ul>
                <p>Predicted ΔG: ${interactions.binding_energy.toFixed(2)} kcal/mol</p>
            </div>
        `);
    }

    displayOffTargets(offTargets) {
        const resultsDiv = $('#off-target-results');
        resultsDiv.empty();

        resultsDiv.append('<h4>Off-Target Analysis</h4>');

        offTargets.forEach(target => {
            const riskClass = target.risk_score > 0.7 ? 'high-risk' :
                            target.risk_score > 0.3 ? 'medium-risk' : 'low-risk';

            resultsDiv.append(`
                <div class="off-target ${riskClass}">
                    <h5>${target.protein_name}</h5>
                    <p>Similarity: ${(target.similarity * 100).toFixed(1)}%</p>
                    <p>Risk Score: ${(target.risk_score * 100).toFixed(1)}%</p>
                    <p>Potential Effects: ${target.effects.join(', ')}</p>
                </div>
            `);
        });
    }
}
