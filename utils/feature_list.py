# utils/feature_list.py

FEATURE_LIST = [
    "protocol_type",          # Protocol used (TCP, UDP, ICMP) — key for probe patterns
    "flag",                   # Connection flag status — helps detect incomplete handshakes
    "destination_port",       # Target port — probes usually sweep many ports
    "flow_duration",          # Time duration of the connection
    "total_forward_packets",  # Packets sent from source
    "total_backward_packets", # Packets sent from destination
    "average_packet_size",    # Average size of packets — scanning often has small fixed sizes
    "flow_bytes_per_s",       # Bytes per second — low but steady for scans
    "fwd_iat_mean",           # Mean time between forward packets
    "bwd_iat_mean"            # Mean time between backward packets
]

# Normalize spacing (prevents accidental mismatches in feature names)
FEATURE_LIST = [f.strip() for f in FEATURE_LIST]