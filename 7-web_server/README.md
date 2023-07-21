# Web server of PLANet Database(PLANet_Uw for now)
## 1. Enviroment
### 1.1 python packges
- build database: `conda env create -f conda.env.web_server.yml`
### 1.2 Install PostgreSQL
- [reference 1](https://github.com/hnlab/handbook/blob/master/3-Materials/Dataset-database/psql.md)
- [reference 2](https://github.com/hnlab/iso_database/tree/xiegang/inhouse_MMP/6_inhouse_MMP/web)
## 2. Create table for postgreSQL
- `cd 1-mk_table`
- `python table.py`

|Table Name|Primary Key|Other Coloumns|Foreign Key|
|-|-|-|-|
|complex|(target_chembl_id, pdbid, compound)|total_sampled_num,calculated_binding_energy, calculated_delta_lig_conform_energy, core_RMSD||
|pdbid_mapped_uniprot_family_chembl_info.csv||pdbid, uniprot_id_for_pdbid, uniprot_id, target_name_from_uniprot, target_name_from_pharos, target_family_from_pharos, target_chembl_id, uniprot_id_from_chembl, target_name_from_chembl, target_type_from_chembl||
|crylig|pdbid|crylig_smiles, crylig_heavy_atom_number||
|compound|compound|compound_smiles, compound_heavy_atom_number, compound_Mw||
|crylig_compound|(pdbid, compound)|similarity, core_heavy_atom_number, different_heavy_atom_number, fix_part_core, core_smarts||
|activity||target_chembl_id, compound, activity_type, activity_relation, activity_value, activity_units, assay_chembl_id, pAffi|assay_chembl_id|
|assay|assay_chembl_id|assay_type||

## 3. Import tables into postgreSQL
- `cd 2-import_table_into_postgres`
### 3.1 Create Role and Database `planet_uw_v1`
- `sudo -u postgres psql`
- useful sql commands
    - `\du`: View all roles
    - `\l`: view all databases
    - `\c databasename`: connet to database
    - `\d`: view all tables of current database
    - `\di`: view all indexes and keys of current database
```sql
CREATE ROLE planet WITH CREATEDB NOLOGIN;

# CREATE ROLE lixl WITH CREATEDB LOGIN IN ROLE planet; # role exists
# CREATE ROLE "user" WITH LOGIN PASSWORD 'user' IN ROLE planet; # role exists
# https://www.postgresql.org/docs/9.0/role-membership.html
GRANT planet TO lixl;
GRANT planet TO "user";
ALTER ROLE lixl WITH PASSWORD 'lixl';

CREATE DATABASE planet_uw_v1;
```
### 3.2 Modify host-based authentication file(`/data/pgdata/data/pg_hba.conf`)
- `sudo cp pg_hba.conf /data/pgdata/data/pg_hba.conf`
- Restart: `sudo systemctl restart postgresql-12`
- Test: `psql -h 192.168.54.19 -U user -d planet_uw_v1`
    - password: `user`
### 3.3 Create tables and Import tables as superuser `postgres`
- copy all 7 `.csv` tables to `/data/pgdata/planet_Uw`
- change the owner of planet_Uw directory: `sudo chown -R postgres:postgres /data/pgdata/planet_Uw`
- `sudo -u postgres psql`
- sql: `\c planet_uw_v1`
- Excute SQL commands like `import_table_into_postgre.sql`
- Change owner from `postgres` to `lixl`: `ALTER TABLE activity OWNER TO lixl;`
- `GRANT SELECT ON TABLE activity TO "user";`
- Test connect in python
```sql
import psycopg2
import psycopg2.extras
conn = psycopg2.connect(dbname='planet_uw_v1', 
                        user='user', 
                        password='user', 
                        host='192.168.54.19')
with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
    cur.execute("SELECT * FROM crylig WHERE pdbid = '6en4';")
    results = [dict(dictrow) for dictrow in cur.fetchall()]   #cur.fetchall() 接收全部的返回结果行；
conn.close()
```
## 4. Backend and frontend files
- `cd 3-web_server`
- `python planet.py`
