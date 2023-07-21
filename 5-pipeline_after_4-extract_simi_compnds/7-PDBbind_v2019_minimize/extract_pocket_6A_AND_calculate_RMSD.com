open rec_h_opt.pdb
addcharge
open cry_lig_opt.pdb
open cry_lig_grepped.pdb
sel #1 z<6
write format mol2 selected 0 rec_addcharge_pocket_6.mol2
del H
rmsd #1 #2
close all
stop
