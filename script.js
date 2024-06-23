document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();
    predict();
});

function predict() {
    // Collect user inputs
    const potential = document.getElementById('potential').value;
    const value_eur = document.getElementById('value_eur').value;
    const wage_eur = document.getElementById('wage_eur').value;
    const passing = document.getElementById('passing').value;
    const dribbling = document.getElementById('dribbling').value;
    const physic = document.getElementById('physic').value;
    const movement_reactions = document.getElementById('movement_reactions').value;
    const mentality_composure = document.getElementById('mentality_composure').value;

    // Prepare data to send to Flask backend
    const data = {
        potential: potential,
        value_eur: value_eur,
        wage_eur: wage_eur,
        passing: passing,
        dribbling: dribbling,
        physic: physic,
        movement_reactions: movement_reactions,
        mentality_composure: mentality_composure
    };

    // Send POST request to Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Display prediction and confidence interval
        const confidenceInterval = data.confidence_interval;
        document.getElementById('prediction').textContent = 'Rating: ' + data.prediction.toFixed(2);
        document.getElementById('confidence').textContent = 'Confidence Interval (90% Confidence):\n' + 
            confidenceInterval[0].toFixed(2) + ' - ' + confidenceInterval[1].toFixed(2);
        document.getElementById('result').style.display = 'block';  // Show result div
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
