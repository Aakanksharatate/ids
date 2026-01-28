// dashboard.js
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const resultBox = document.getElementById("result");
    const predictionOutput = document.getElementById("predictionOutput");
    const confidenceOutput = document.getElementById("confidenceOutput");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        // Collect input values
        const data = {
            protocol_type: document.getElementById("protocol_type").value.trim(),
            flag: document.getElementById("flag").value.trim(),
            destination_port: parseInt(document.getElementById("destination_port").value),
            flow_duration: parseFloat(document.getElementById("flow_duration").value),
            total_forward_packets: parseInt(document.getElementById("total_forward_packets").value),
            total_backward_packets: parseInt(document.getElementById("total_backward_packets").value),
            average_packet_size: parseFloat(document.getElementById("average_packet_size").value),
            flow_bytes_per_s: parseFloat(document.getElementById("flow_bytes_per_s").value),
            fwd_iat_mean: parseFloat(document.getElementById("fwd_iat_mean").value),
            bwd_iat_mean: parseFloat(document.getElementById("bwd_iat_mean").value)
        };

        // API call to Flask backend
        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await response.json();

          if (response.ok && result.prediction) {
    predictionOutput.textContent = result.prediction;
    confidenceOutput.textContent = result.confidence || "N/A";
    resultBox.style.display = "block";

    if (result.prediction.toLowerCase() === "attack") {
        resultBox.classList.remove("normal");
        resultBox.classList.add("attack");
    } else {
        resultBox.classList.remove("attack");
        resultBox.classList.add("normal");
    }
} else {
    console.error("Unexpected result from backend:", result);
    alert("⚠️ Error: " + (result.error || "Unexpected response format."));
}
        } catch (error) {
            alert("❌ Unable to connect to backend. Please check if Flask is running.");
            console.error(error);
        }
    });
});
