import papermill as pm

FOLDERS = [
    "AIRWAY_HYPERREACTIVITY", "ARTHRITIS", "CROHNS", 
    "DIABETES_II", "HEART_FAILURE", "HIV", "NEOPLASM_BREAST"
]

for folder in FOLDERS:
    folder_name = "DGIDB_" + folder + "/"
    pm.execute_notebook(
        'Unidirectional-Multilayer copy.ipynb',
        f'output_notebooks/{folder}.ipynb',
        parameters={'FOLDER': folder_name}
    )
