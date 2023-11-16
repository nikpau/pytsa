"""
Target ship extraction agent (:mod:`pytsa.tsea`)
=============================

This module contains provides twofold functionality:
    1. Extract target ship trajectories from raw AIS data
    2. Query for target ships around a given position and time

Both functionalities rely on the :class:`SearchAgent` class, 
which is initialized with the AIS data and a geographic bounding box.

AIS DATA ------------------------------
The AIS data are expected to be decoded CSV files, split up into two files:
    1. Dynamic data (messages 1,2,3,18)
    2. Static data (message 5)
    
If your messages are not decoded yet, please refer to the
:mod:`pytsa.decode` module.
"""