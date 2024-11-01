$(document).ready(function() {
    console.log('Initializing ProteinFlex visualization...');

    // Initialize the protein visualizer with proper error handling
    let proteinVisualizer;
    try {
        proteinVisualizer = new ProteinVisualizer('viewer-container', 'contact-map');
        console.log('Viewer initialized:', proteinVisualizer.viewer !== null);
    } catch (error) {
        console.error('Failed to initialize visualizer:', error);
        $('#viewer-container').html('<div class="error-message">Failed to initialize 3D viewer. Please refresh the page.</div>');
        return;
    }

    // Set up sequence input validation
    $('#sequence-input').on('input', function() {
        const sequence = $(this).val().toUpperCase();
        const validSequence = sequence.replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
        $(this).val(validSequence);

        // Show warning for large sequences
        if (validSequence.length > 500) {
            $('#sequence-warning').text('Warning: Large sequences may take longer to process').show();
        } else {
            $('#sequence-warning').hide();
        }
    });

    // Set up prediction button with loading state
    $('#predict-button').click(function() {
        const button = $(this);
        const originalText = button.text();
        const sequence = $('#sequence-input').val();

        // Validate sequence
        if (!sequence) {
            alert('Please enter a protein sequence');
            return;
        }

        if (sequence.length > 1000) {
            if (!confirm('Processing large sequences may take several minutes. Continue?')) {
                return;
            }
        }

        // Show loading state with progress indicator
        button.prop('disabled', true).text('Processing...');
        $('.loading-spinner').addClass('active');
        $('#progress-indicator').show();

        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({sequence: sequence}),
            timeout: 300000, // 5-minute timeout for large sequences
            success: function(response) {
                try {
                    // Update progress indicator
                    $('#progress-indicator').text('Visualizing structure...');

                    // Use requestAnimationFrame for smoother visualization
                    requestAnimationFrame(() => {
                        proteinVisualizer.visualizeStructure(response.pdb_string, response.confidence_score);
                        proteinVisualizer.updateContactMap(response.contact_map);
                        $('#analysis-results').text(response.description);
                        console.log('Visualization updated successfully');
                    });
                } catch (error) {
                    console.error('Error updating visualization:', error);
                    alert('Error visualizing protein structure: ' + error.message);
                }
            },
            error: function(xhr, status, error) {
                console.error('Prediction error:', error);
                if (status === 'timeout') {
                    alert('Request timed out. Please try a smaller sequence or try again later.');
                } else {
                    alert('Error predicting protein structure: ' + error);
                }
            },
            complete: function() {
                button.prop('disabled', false).text(originalText);
                $('.loading-spinner').removeClass('active');
                $('#progress-indicator').hide();
            }
        });
    });

    // Set up control button handlers
    $('#reset-view').click(() => proteinVisualizer.resetView());
    $('#toggle-surface').click(() => proteinVisualizer.toggleSurface());
    $('#toggle-sidebar').click(function() {
        $('#sidebar').toggleClass('collapsed');
        $('#visualization-section').toggleClass('expanded');
        if (proteinVisualizer.viewer) {
            proteinVisualizer.viewer.resize();
        }
    });

    // Handle protein questions
    $('#ask-question').click(function() {
        const sequence = $('#sequence-input').val();
        const question = $('#question-input').val();
        if (!sequence || !question) {
            alert('Please enter both a sequence and a question');
            return;
        }

        $('.loading-spinner').addClass('active');
        $.ajax({
            url: '/ask_question',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                sequence: sequence,
                question: question
            }),
            success: function(response) {
                $('.loading-spinner').removeClass('active');
                $('#answer-box').html(response.answer);
                if (response.confidence) {
                    $('#answer-confidence').text(`Confidence: ${(response.confidence * 100).toFixed(1)}%`);
                }
            },
            error: function(error) {
                $('.loading-spinner').removeClass('active');
                console.error('Error:', error);
                alert('Error processing question. Please try again.');
            }
        });
    });

    // Handle mutation analysis
    $('#analyze-mutation').click(function() {
        const sequence = $('#sequence-input').val();
        const mutation = $('#mutation-input').val();
        if (!sequence || !mutation) {
            alert('Please enter both a sequence and a mutation');
            return;
        }

        $('.loading-spinner').addClass('active');
        $.ajax({
            url: '/analyze_mutation',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                sequence: sequence,
                mutation: mutation
            }),
            success: function(response) {
                $('.loading-spinner').removeClass('active');
                displayMutationResults(response);
                if (response.overall_impact) {
                    proteinVisualizer.highlightMutation(response.mutation, response.overall_impact);
                }
            },
            error: function(error) {
                $('.loading-spinner').removeClass('active');
                console.error('Error:', error);
                alert('Error analyzing mutation. Please try again.');
            }
        });
    });

    function displayMutationResults(results) {
        const resultsDiv = $('#mutation-results');
        resultsDiv.empty();

        const impactClass = results.overall_impact > 0.7 ? 'impact-high' :
                          results.overall_impact > 0.3 ? 'impact-medium' : 'impact-low';

        const html = `
            <div class="mutation-item">
                <div class="mutation-header">
                    <span class="confidence-indicator ${impactClass}"></span>
                    <strong>Mutation: ${results.mutation}</strong>
                </div>
                <div class="mutation-details">
                    <p>Stability Impact: ${(results.stability_impact * 100).toFixed(1)}%</p>
                    <p>Structural Impact: ${(results.structural_impact * 100).toFixed(1)}%</p>
                    <p>Functional Impact: ${(results.functional_impact * 100).toFixed(1)}%</p>
                    <p>Overall Impact: ${(results.overall_impact * 100).toFixed(1)}%</p>
                    <p>Confidence: ${(results.confidence * 100).toFixed(1)}%</p>
                </div>
            </div>
        `;
        resultsDiv.html(html);
    }

    // Sample protein sequence for testing
    const sampleSequence = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR';
    $('#sequence-input').val(sampleSequence);
});
