# Tessellates the obstacle CAD (STEP AP242) into the STL visual meshes the
# obstacle-course world references.  Kept as a container because the only
# maintained OpenCASCADE bindings that pip-install cleanly (cascadio) need a
# Python the Windows and Orin hosts do not both have.
#
#   docker build -f jetson/scripts/step_to_stl.Dockerfile -t cfr-step-to-stl jetson/scripts
#   docker run --rm -v "<cad dir>:/cad:ro" -v "<repo>/jetson:/jetson" cfr-step-to-stl \
#       /jetson/scripts/step_to_stl.py /cad /jetson/cfr_arduino_bridge/meshes
FROM python:3.12-slim

RUN pip install --no-cache-dir cascadio==0.1.1 trimesh==5.1.0 numpy fast-simplification

ENTRYPOINT ["python3"]
