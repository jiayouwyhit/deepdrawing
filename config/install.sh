pip uninstall torch-scatter
pip uninstall torch-sparse
pip uninstall torch-cluster
pip uninstall torch-spline-conv
pip uninstall torch-geometric
conda install pytorch=1.0.0 cudatoolkit=9.0 -c pytorch 
pip install --no-cache-dir torch-scatter==1.1.2
pip install --no-cache-dir torch-sparse==0.2.4
pip install --no-cache-dir torch-cluster==1.2.4
pip install --no-cache-dir torch-spline-conv==1.0.6
pip install --no-cache-dir torch-geometric==1.0.3
