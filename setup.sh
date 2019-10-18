apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
  	curl \
	ca-certificates \
	libjpeg-dev \
	libpng-dev
rm -rf /var/lib/apt/lists/*

PYTORCH_VERSION=1.2
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p /opt/conda
rm ~/miniconda.sh
conda install -y openblas numpy
conda install -y -c bioconda google-sparsehash
conda install -y pytorch=$PYTORCH_VERSION torchvision -c pytorch
conda clean -ya
cd
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
cd
rm -r MinkowskiEngine
pip install open3d
apt-get -y install mesa-common-dev libglu1-mesa-dev
