document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const resultBox = document.getElementById("result");
    const predictionOutput = document.getElementById("predictionOutput");
    const confidenceOutput = document.getElementById("confidenceOutput");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        // reset UI (IMPORTANT)
        resultBox.style.display = "none";

        const data = {
            protocol_type: document.getElementById("protocol_type").value.trim(),
            flag: document.getElementById("flag").value.trim(),
            destination_port: Number(document.getElementById("destination_port").value),
            flow_duration: Number(document.getElementById("flow_duration").value),
            total_forward_packets: Number(document.getElementById("total_forward_packets").value),
            total_backward_packets: Number(document.getElementById("total_backward_packets").value),
            average_packet_size: Number(document.getElementById("average_packet_size").value),
            flow_bytes_per_s: Number(document.getElementById("flow_bytes_per_s").value),
            fwd_iat_mean: Number(document.getElementById("fwd_iat_mean").value),
            bwd_iat_mean: Number(document.getElementById("bwd_iat_mean").value)
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok && result.prediction) {
                predictionOutput.textContent = result.prediction;
                confidenceOutput.textContent = result.confidence ?? "N/A";
                resultBox.style.display = "block";

                if (result.prediction.toLowerCase() === "attack") {
                    resultBox.classList.remove("normal");
                    resultBox.classList.add("attack");
                } else {
                    resultBox.classList.remove("attack");
                    resultBox.classList.add("normal");
                }
            } else {
                alert("Error: " + (result.error || "Unexpected response"));
            }
        } catch (error) {
            console.error(error);
            alert("Unable to connect to backend");
        }
    });
});
