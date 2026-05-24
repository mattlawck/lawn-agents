"""External-I/O and retrieval agents.

Every module in this package is responsible for one external boundary
(NWS, AWDB/SCAN, Drought Monitor, the local knowledge index, or the
research subagent). Each fails closed to `None` and never raises past
its public function boundary.
"""
