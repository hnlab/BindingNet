CREATE TABLE crylig(
    pdbid VARCHAR(4) NOT NULL,
    crylig_smiles VARCHAR(1000) NOT NULL,
    crylig_heavy_atom_number INT NOT NULL,
    PRIMARY KEY (pdbid)
);
COPY crylig FROM '/data/pgdata/planet_Uw/crylig.csv' DELIMITER E'\t' CSV HEADER; -- COPY 5908

-- CREATE TABLE target_info(
--     target_chembl_id VARCHAR(13) NOT NULL,
--     uniprot_id VARCHAR(6) NOT NULL,
--     target_name_from_uniprot VARCHAR(1200) NOT NULL,
--     target_name_from_pharos VARCHAR(100),
--     target_family_from_pharos VARCHAR(50)
-- );
-- COPY target_info FROM '/data/pgdata/planet_Uw/target.csv' DELIMITER E'\t' CSV HEADER; -- COPY 813

CREATE TABLE compound(
    compound VARCHAR(13) NOT NULL,
    compound_smiles VARCHAR(250) NOT NULL,
    compound_heavy_atom_number INT NOT NULL,
    compound_Mw NUMERIC NOT NULL,
    PRIMARY KEY (compound)
);
COPY compound FROM '/data/pgdata/planet_Uw/compound.csv' DELIMITER E'\t' CSV HEADER; -- COPY 65924

CREATE TABLE assay(
    assay_chembl_id VARCHAR(13) NOT NULL,
    assay_type VARCHAR(1) NOT NULL,
    PRIMARY KEY (assay_chembl_id)
);
COPY assay FROM '/data/pgdata/planet_Uw/assay.csv' DELIMITER E'\t' CSV HEADER; -- COPY 16434

CREATE TABLE activity(
    target_chembl_id VARCHAR(13) NOT NULL,
    compound VARCHAR(13) NOT NULL,
    activity_type VARCHAR(4) NOT NULL,
    activity_relation VARCHAR(1) NOT NULL,
    activity_value NUMERIC NOT NULL,
    activity_units VARCHAR(2) NOT NULL,
    assay_chembl_id VARCHAR(13) NOT NULL,
    pAffi NUMERIC NOT NULL,
    FOREIGN KEY(assay_chembl_id) REFERENCES assay(assay_chembl_id)
);
COPY activity FROM '/data/pgdata/planet_Uw/activity.csv' DELIMITER E'\t' CSV HEADER; -- COPY 105920

CREATE TABLE complex(
    target_chembl_id VARCHAR(13) NOT NULL,
    pdbid VARCHAR(4) NOT NULL,
    compound VARCHAR(13) NOT NULL,
    total_sampled_num INT NOT NULL,
    calculated_binding_energy NUMERIC NOT NULL,
    calculated_delta_lig_conform_energy NUMERIC NOT NULL,
    core_RMSD NUMERIC NOT NULL,
    PRIMARY KEY (target_chembl_id, pdbid, compound)
);
COPY complex FROM '/data/pgdata/planet_Uw/complex.csv' DELIMITER E'\t' CSV HEADER; -- COPY 69826

CREATE TABLE crylig_compound(
    pdbid VARCHAR(4) NOT NULL,
    compound VARCHAR(13) NOT NULL,
    similarity NUMERIC NOT NULL,
    core_heavy_atom_number INT NOT NULL,
    different_heavy_atom_number INT NOT NULL,
    fix_part_core VARCHAR(3) NOT NULL,
    core_smarts VARCHAR(600) NOT NULL,
    PRIMARY KEY (pdbid, compound)
);
COPY crylig_compound FROM '/data/pgdata/planet_Uw/crylig_compound.csv' DELIMITER E'\t' CSV HEADER; -- COPY 69400

CREATE TABLE pdbid_mapped_uniprot_family_chembl_info(
    pdbid VARCHAR(4) NOT NULL,
    uniprot_id_for_pdbid VARCHAR(200) NOT NULL,
    uniprot_id VARCHAR(6) NOT NULL,
    target_name_from_uniprot VARCHAR(1200) NOT NULL,
    target_name_from_pharos VARCHAR(100),
    target_family_from_pharos VARCHAR(50),
    target_chembl_id VARCHAR(13) NOT NULL,
    uniprot_id_from_chembl VARCHAR(200) NOT NULL,
    target_name_from_chembl VARCHAR(90) NOT NULL,
    target_type_from_chembl VARCHAR(30) NOT NULL
);
COPY pdbid_mapped_uniprot_family_chembl_info FROM '/data/pgdata/planet_Uw/pdbid_mapped_uniprot_family_chembl_info.csv' DELIMITER E'\t' CSV HEADER; -- COPY 6032


-- CREATE TABLE target_info_from_chembl(
--     target_chembl_id VARCHAR(13) NOT NULL,
--     uniprot_id_from_chembl VARCHAR(200) NOT NULL,
--     target_name_from_chembl VARCHAR(90) NOT NULL,
--     target_type_from_chembl VARCHAR(30) NOT NULL,
--     PRIMARY KEY (target_chembl_id)
-- );
-- COPY target_info_from_chembl FROM '/data/pgdata/planet_Uw/target_info_from_chembl.csv' DELIMITER E'\t' CSV HEADER; -- COPY 803

-- CREATE TABLE pdbid_to_uniprot_id(
--     pdbid VARCHAR(4) NOT NULL,
--     uniprot_id_for_pdbid VARCHAR(200) NOT NULL,
--     PRIMARY KEY (pdbid)
-- );
-- COPY pdbid_to_uniprot_id FROM '/data/pgdata/planet_Uw/pdbid_to_uniprot_id.csv' DELIMITER E'\t' CSV HEADER; -- COPY 5908

-- SELECT uniprot_id FROM complex INNER JOIN target ON complex.target_chembl_id = target.target_chembl_id WHERE complex.target_chembl_id = CHEMBL301;