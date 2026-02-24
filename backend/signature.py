# backend/signature_rules.py

def check_signature(data):
    """
    data: dict of input features
    return: (is_attack: bool, reason: str)
    """

    # ---- Rule 1: Port Scan ----
    if (
        data["flow_duration"] < 100 and
        data["total_forward_packets"] > 20
    ):
        return True, "Port Scan Detected"

    # ---- Rule 2: Suspicious TCP Flags ----
    if data["flag"] in ["REJ", "RSTO", "RSTR", "S0"]:
        return True, "Suspicious TCP Flag"

    # ---- Rule 3: DoS-like traffic ----
    if (
        data["destination_port"] in [80, 443] and
        data["flow_bytes_per_s"] > 1_000_000
    ):
        return True, "High Traffic Rate (Possible DoS)"

    # ---- Rule 4: One-way packet flood ----
    if (
        data["total_forward_packets"] > 50 and
        data["total_backward_packets"] == 0
    ):
        return True, "One-Way Packet Flood"

    return False, None
