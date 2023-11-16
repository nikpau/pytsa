"""
Target ship extraction agent (:mod:`pytsa.tsea`)
=============================

This module contains provides twofold functionality:
    1. Extract target ships from AIS data
    2. Query for target ships around a given position and time

Both functionalities rely on the :class:`SearchAgent` class, 
which is initialized with the AIS data and a geographic bounding box.

AIS DATA ------------------------------
The AIS data are expected to be CSV files, split up into two files:
    1. Dynamic data (messages 1,2,3,18)
    2. Static data (message 5)
"""