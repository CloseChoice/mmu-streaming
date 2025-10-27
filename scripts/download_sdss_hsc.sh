wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/hsc/pdr3_dud_22.5/healpix=1175/
wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/hsc/hsc.py
wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss/healpix=1175/
wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss.py

# Check if .py files were downloaded, if not copy from additional_dataset_files
echo "Checking for .py files..."

if [ ! -f "data/MultimodalUniverse/v1/hsc/hsc.py" ]; then
    echo "hsc.py not found, copying from additional_dataset_files..."
    mkdir -p data/MultimodalUniverse/v1/hsc
    cp additional_dataset_files/hsc.py data/MultimodalUniverse/v1/hsc/hsc.py
else
    echo "hsc.py found"
fi

if [ ! -f "data/MultimodalUniverse/v1/sdss/sdss.py" ]; then
    echo "sdss.py not found, copying from additional_dataset_files..."
    mkdir -p data/MultimodalUniverse/v1/sdss
    cp additional_dataset_files/sdss.py data/MultimodalUniverse/v1/sdss/sdss.py
else
    echo "sdss.py found"
fi

echo "Download complete!"
