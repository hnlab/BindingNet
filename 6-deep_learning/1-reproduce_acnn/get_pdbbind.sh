test -d pdbbind || mkdir pdbbind
cd pdbbind

echo "Pulling pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz
echo "Extracting pdbbind structures"
tar -zxvf pdbbind_v2015.tar.gz

cd v2015

# same index location as pdbbind v2018
test -d index || mkdir index
cp INDEX* index

# convert ligands from mol2 to pdb using openbabel
# rdkit read mol2, valid ligands: 10443/11918
# rdkit read pdb, valid ligands: 11841/11918
ls */*ligand.mol2 | parallel obabel -imol2 {} -opdb -O{.}.pdb