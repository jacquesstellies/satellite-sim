ROOT_DIR=git rev-parse --show-toplevel
cd $ROOT_DIR
source .venv/bin/activate
mkdir -p ../data_logs
pip install -r requirements.txt
