from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import psycopg2
import psycopg2.extras
import pandas as pd
from pager import Pagination
from rdkit.Chem import PandasTools

app = Flask(__name__, static_url_path="", static_folder="./")
app.secret_key = 'huanglab'


def get_compnd_count_data(query_type, query_txt):
    sql_select = ["complex.target_chembl_id", "uniprot_id", "target_name_from_uniprot", "target_name_from_pharos", "target_family_from_pharos", "pdbid", "compound"]
    sql_join = " FROM complex\
                INNER JOIN target_info ON complex.target_chembl_id = target_info.target_chembl_id"
    if query_type in 'target_chembl_id':
        sql_where = f"complex.{query_type} = '{query_txt}'"
    elif query_type == 'target_uniprot_id':
        sql_where = f"uniprot_id = '{query_txt}'"
    elif query_type in 'pdbid':
        sql_where = f"{query_type} = '{query_txt}'"
    conn = psycopg2.connect(dbname='planet_uw_v1', 
                            user='user', 
                            password='user', 
                            host='192.168.54.19')
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f"SELECT {','.join(sql_select)} {sql_join} WHERE {sql_where};")
        results = [dict(dictrow) for dictrow in cur.fetchall()]   #cur.fetchall() 接收全部的返回结果行；
    conn.close()
    results_df = pd.DataFrame(results)
    return results_df


def get_compound_info(target_chembl_id, pdbid):
    sql_select = ["complex.target_chembl_id", 
                "complex.pdbid", "crylig_smiles", 
                "complex.compound", "compound_smiles",
                "similarity", 
                "activity_type", "activity_relation", "activity_value", "activity_units", "pAffi", "activity.assay_chembl_id", 
                "assay_type", 
                "total_sampled_num", "calculated_binding_energy", "calculated_delta_lig_conform_energy", "core_RMSD",
                "fix_part_core", "core_heavy_atom_number", ]
    sql_join = " FROM complex\
                INNER JOIN activity ON complex.compound = activity.compound AND complex.target_chembl_id = activity.target_chembl_id\
                INNER JOIN assay ON assay.assay_chembl_id = activity.assay_chembl_id\
                INNER JOIN compound ON complex.compound = compound.compound\
                INNER JOIN crylig_compound ON complex.pdbid = crylig_compound.pdbid AND complex.compound = crylig_compound.compound\
                INNER JOIN crylig ON complex.pdbid = crylig.pdbid"

    conn = psycopg2.connect(dbname='planet_uw_v1', 
                            user='user', 
                            password='user', 
                            host='192.168.54.19')
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f"SELECT {','.join(sql_select)} {sql_join} WHERE complex.target_chembl_id = '{target_chembl_id}' AND complex.pdbid = '{pdbid}';")
        results = [dict(dictrow) for dictrow in cur.fetchall()]   #cur.fetchall() 接收全部的返回结果行；
    conn.close()
    results_df = pd.DataFrame(results)

    results_title_df = results_df[["target_chembl_id", "pdbid", "crylig_smiles"]].copy()
    results_title_df.drop_duplicates(inplace=True)
    PandasTools.AddMoleculeColumnToFrame(results_title_df, "crylig_smiles", "crylig_mol")

    results_df['activity'] = [f'{row.activity_type} {row.activity_relation} {row.activity_value} {row.activity_units}' for row in results_df.itertuples()]
    PandasTools.AddMoleculeColumnToFrame(results_df, "compound_smiles", "compound_mol")
    results_recolum_df = results_df[["assay_chembl_id", "assay_type",
            "compound", "compound_smiles", "compound_mol",
            "similarity", 
            "activity", "paffi",
            "total_sampled_num", "calculated_binding_energy", "calculated_delta_lig_conform_energy", "core_rmsd",
            "fix_part_core", "core_heavy_atom_number"]].copy()
    results_recolum_df.rename(columns={"assay_chembl_id":"Assay ChEMBL ID", "assay_type":"Assay Type", "compound": "Compound", "compound_smiles": "Compound Smiles", "compound_mol": "Compound Mol", "similarity": "Similarity", "activity": "Activity", "paffi": "pAffi", "total_sampled_num": "Total Sampled Num", "calculated_binding_energy": "Calculated Binding Energy (kcal/mol)", "calculated_delta_lig_conform_energy": "Calculated delta compnd comform Energy (kcal/mol)", "core_rmsd": "core RMSD (Å)", "fix_part_core": "Fix Part of core", "core_heavy_atom_number": "core Heavy Atom Num"}, inplace=True)
    results_recolum_reindex_df = results_recolum_df.set_index(["Assay ChEMBL ID", "Assay Type"]).sort_index()
    return results_title_df, results_recolum_reindex_df


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('homepage.html')
    else:
        query_type = request.form.get('query_type')
        query_txt = request.form.get('query_txt')
        if query_type == "" or query_txt == "":
            return render_template('homepage.html')
        #print(query_type,query_txt)
        # if smirks != "":
        #     state = judge_smirks(smirks)
        #     if state == "True":
        #         flash("Wait for a moment...")
        #     else:
        #         return render_template('homepage.html')
        # global all_df
        # compnd_count_df = get_compnd_count_data(query_type=query_type,query_txt=query_txt)
        # if compnd_count_df.shape[0]==0:
        #     return render_template('homepage.html')
        return redirect(url_for("query", query_type=query_type, query_txt=query_txt))


@app.route('/query',methods=['GET','POST'])
def query():
    query_type = request.args.get("query_type")
    query_txt = request.args.get("query_txt")
    compnd_count_df = get_compnd_count_data(query_type=query_type,query_txt=query_txt)
    if compnd_count_df.shape[0]==0:
        return render_template('homepage.html')

    page = request.args.get("page",default= 1)
    size = int(request.args.get("size",default= 10))
    # if request.method == 'POST':
        # order_column = request.form.get("orderColumn")
        # order = request.form.get("orderBy")
        # filter_column = request.form.get("filterColumn")
        # name = request.form.get("filterBy")
        #print(column,order,filter_column,name)
        # df_view = view_dataframe(order_column,order,filter_column,name)
    index_colums = ["target_chembl_id", "uniprot_id", "target_name_from_uniprot", "target_name_from_pharos", "target_family_from_pharos", "pdbid"]
    df_view = compnd_count_df.drop_duplicates().groupby(index_colums).count()
    df_view.rename(columns={"compound": "Num of similar compounds"}, inplace=True)
    pager_obj = Pagination(page, df_view.shape[0], request.path, request.args, per_page_count=size)
    table_html = df_view[pager_obj.start:pager_obj.end].to_html(classes = "table",border=0,render_links=True,escape=False)
    phtml = pager_obj.page_html()
    current_path = url_for("query",query_type=query_type, query_txt=query_txt)
    return render_template('query.html',
                            query_type=query_type,
                            query_txt=query_txt,
                            # columns = list(df_view.columns),
                            table_html=table_html,
                            phtml=phtml,
                            current_path=current_path,
                            size=size,
                            nrow=df_view.shape[0],)


@app.route('/structure/<target_chembl_id>/<pdbid>/') 
def get_structure(target_chembl_id, pdbid):
    data_dir = '/home/lixl/data/from_x254/'
    pdbbind_dir = f'{data_dir}/PDBbind_v2019/general_structure_only/{pdbid}'
    # urls = [send_from_directory(f'{pdbbind_dir}', f'{pdbid}_protein.pdb')]
    # urls = urls + [send_from_directory(f'{pdbbind_dir}', f'{pdbid}_ligand.mol2')]

    # compounds = list(all_df[(all_df['target_chembl_id']==target_chembl_id) & (all_df['pdbid']==pdbid)]['compound'])
    # target_pdb_dir = f'{data_dir}/ChEMBL-scaffold/v2019_dataset/web_client_{target_chembl_id}/{target_chembl_id}_{pdbid}'
    # rec_opt_url = send_from_directory(f'{target_pdb_dir}/rec_opt', 'rec_h_opt.pdb')
    # urls = urls + rec_opt_url
    # urls = urls + [send_from_directory(f'{target_pdb_dir}/{compnd}', 'compound.pdb') for compnd in compounds]
    return send_from_directory(f'{pdbbind_dir}', f'{pdbid}_protein.pdb')


@app.route("/view/<target_chembl_id>/<pdbid>",methods=['GET','POST'])
def view(target_chembl_id, pdbid):
    return render_template('viewer.html', 
                            target_chembl_id=target_chembl_id,
                            pdbid=pdbid)


@app.route("/final_table/<target_chembl_id>/<pdbid>/",methods=['GET','POST'])
def final_table(target_chembl_id, pdbid):
    pdbid = pdbid[:4]
    results_title_df, results_recolum_reindex_df = get_compound_info(target_chembl_id, pdbid)
    if results_recolum_reindex_df.shape[0]==0:
        return render_template('homepage.html')
    title_table_html = results_title_df.to_html(classes = "table",border=0,render_links=True,escape=False)

    page = request.args.get("page",default= 1)
    size = int(request.args.get("size",default= 10))
    pager_obj = Pagination(page, results_recolum_reindex_df.shape[0], request.path, request.args, per_page_count=size)
    table_html = results_recolum_reindex_df[pager_obj.start:pager_obj.end].to_html(classes = "table",border=0,render_links=True,escape=False)
    phtml = pager_obj.page_html()
    current_path = url_for("final_table", target_chembl_id=target_chembl_id, pdbid=pdbid)
    return render_template('final_table.html',
                            target_chembl_id=target_chembl_id,
                            pdbid=pdbid,
                            title_table_html = title_table_html,
                            # columns = list(df_view.columns),
                            table_html=table_html,
                            phtml=phtml,
                            current_path=current_path,
                            size=size,
                            nrow=results_recolum_reindex_df.shape[0],)

@app.route('/query_show/<target_chembl_id>/<pdbid>/')
def query_show(target_chembl_id, pdbid):
    return render_template('query_show.html', 
                            target_chembl_id=target_chembl_id,
                            pdbid=pdbid)

if __name__ == "__main__":
    app.run(debug=True,host='192.168.54.19',port=1282)
