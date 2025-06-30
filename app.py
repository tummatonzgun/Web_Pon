import io
import re
import os, time, threading
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from scipy.stats import zscore
from flask import flash, redirect, url_for
##import dropbox
from flask import Flask, send_from_directory
import glob
from io import BytesIO
import urllib.parse 
from flask import send_file
import socket
import requests

# === ตั้งค่า ===
##DROPBOX_ACCESS_TOKEN = "<YOUR_ACCESS_TOKEN>"  # <-- ไม่ใช้แล้ว
UPLOAD_FOLDER = 'static/graphs/js'
DATA_FOLDER = './data'
FRAMESTOCK_FOLDER = 'uploads'
PACKTYPE_FOLDER = 'packtype'
PACKAGECODE_FOLFER = 'package code'
PACKAGEANDFRAMSTOCK = 'package and frame stock'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(FRAMESTOCK_FOLDER, exist_ok=True)
os.makedirs(PACKAGECODE_FOLFER, exist_ok=True)

app = Flask(__name__)
##dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
app.secret_key = '12345'

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def apply_zscore(df):
    col_map = {col.lower(): col for col in df.columns}
    if 'uph' not in col_map:
        raise KeyError("ไม่พบคอลัมน์ UPH ในข้อมูล")
    uph_col = col_map['uph']

    mean = df[uph_col].mean()
    std = df[uph_col].std()
    if std == 0:
        return df  # ถ้า std = 0, return ค่าเดิม
    z_scores = (df[uph_col] - mean) / std
    filtered = df[(z_scores >= -3) & (z_scores <= 3)].copy()
    filtered['Outlier_Method'] = 'Z-Score ±3'
    return filtered

def has_outlier(df):
    col_map = {col.lower(): col for col in df.columns}
    if 'uph' not in col_map:
        raise KeyError("ไม่พบคอลัมน์ UPH ในข้อมูล")
    uph_col = col_map['uph']

    Q1 = df[uph_col].quantile(0.25)
    Q3 = df[uph_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((df[uph_col] < lower) | (df[uph_col] > upper)).sum() > 0

def apply_iqr(df):
    """Apply IQR Method one time."""
    col_map = {col.lower(): col for col in df.columns}
    if 'uph' not in col_map:
        raise KeyError("ไม่พบคอลัมน์ UPH ในข้อมูล")
    uph_col = col_map['uph']

    Q1 = df[uph_col].quantile(0.25)
    Q3 = df[uph_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered = df[(df[uph_col] >= lower) & (df[uph_col] <= upper)].copy()
    filtered['Outlier_Method'] = 'IQR'
    return filtered

def remove_outliers_auto(df_model, max_iter=20):
    col_map = {col.lower(): col for col in df_model.columns}
    if 'uph' not in col_map:
        raise KeyError("ไม่พบคอลัมน์ UPH ในข้อมูล")
    uph_col = col_map['uph']

    df_model[uph_col] = pd.to_numeric(df_model[uph_col], errors='coerce')
    df_model = df_model.dropna(subset=[uph_col])

    if len(df_model) < 15:  
        df_model['Outlier_Method'] = 'ไม่ตัด (ข้อมูลน้อย)'
        return df_model

    current_df = df_model.copy()

    for i in range(max_iter):
        print(f"=== รอบที่ {i+1} ===")
        z_df = apply_zscore(current_df)
        if not has_outlier(z_df):
            z_df['Outlier_Method'] = f'Z-Score Loop ×{i+1}'
            return z_df

        iqr_df = apply_iqr(z_df)
        if not has_outlier(iqr_df):
            iqr_df['Outlier_Method'] = f'IQR Loop ×{i+1}'
            return iqr_df

        current_df = iqr_df

    current_df['Outlier_Method'] = f'IQR-Z-Score Loop ×{max_iter}+'
    return current_df

def remove_outliers(df):
    col_map = {col.lower(): col for col in df.columns}

    model_col = None
    if 'machine model' in col_map:
        model_col = col_map['machine model']
    elif 'machine_model' in col_map:
        model_col = col_map['machine_model']
    else:
        raise KeyError("ไม่พบคอลัมน์ Machine Model หรือ Machine_Model ในข้อมูล")

    return pd.concat([
        remove_outliers_auto(df[df[model_col] == model])
        for model in df[model_col].unique()
    ])

def load_data_by_type(filetype_keyword):
    files = [f for f in os.listdir(DATA_FOLDER) 
             if f.lower().endswith(('.csv', '.xlsx')) and filetype_keyword.lower() in f.lower() and not f.startswith('~$')]

    combined_df = pd.DataFrame()

    for f in files:
        file_path = os.path.join(DATA_FOLDER, f)
        try:
            if f.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                df = pd.read_excel(file_path)
            df['Source_File'] = f
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")

    return combined_df

def process_uph_data(filetype='wb', filepath=None):   
    try:
        if filepath:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, encoding="utf-8-sig")
            else:
                df = pd.read_excel(filepath)
        else:
            df = load_data_by_type(filetype)

        if df.empty:
            return pd.DataFrame()

        df.columns = df.columns.str.strip().str.lower()
        df['uph'] = pd.to_numeric(df['uph'], errors='coerce')
        df.dropna(subset=['uph', 'machine model', 'Machine_Model', 'bom_no'], inplace=True) ####ต้องเพิ่มอีก
        df['bom_no'] = df['bom_no'].astype(str)

        df_clean = remove_outliers(df)
        return df_clean

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

##def download_csv_files_from_dropbox():  ######serverเพิ่มdata DAกับ PNP
    folder_path = '/server'
    try:
        entries = dbx.files_list_folder(folder_path).entries
        for entry in entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith(('.csv', '.xlsx')):
                local_path = os.path.join("temp", entry.name)
                with open(local_path, "wb") as f:
                    metadata, res = dbx.files_download(path=entry.path_lower)
                    f.write(res.content)
                    print(f"✅ Downloaded: {entry.name}")
    except dropbox.exceptions.ApiError as e:
        print(f"❌ Dropbox API Error: {e}")
    except Exception as e:
        print(f"❌ General Error: {e}")

def load_all_nobump_data():
    print(f"📂 ไฟล์ในโฟลเดอร์ temp: {os.listdir('temp')}")
    
    files = [f for f in os.listdir('temp') 
             if ('wire' in f.lower() or 'data' in f.lower() or 'pnp' in f.lower()) 
             and f.endswith(('.xlsx', '.xls')) 
             and not f.startswith('~$')]

    print(f"🔎 ไฟล์ที่ตรงเงื่อนไข: {files}")

    nobump_data = []
    skipped_files = []

    for file in files:
        try:
            file_path = os.path.join('temp', file)
            print(f"🔍 กำลังโหลดไฟล์: {file}")
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip().str.upper()

            print(f"📄 คอลัมน์ในไฟล์ '{file}': {df.columns.tolist()}")

            if all(col in df.columns for col in ['BOM_NO', 'NO_BUMP']):
                if 'NUMBER_REQUIRED' in df.columns:
                    df_use = df[['BOM_NO', 'NO_BUMP', 'NUMBER_REQUIRED']]
                elif 'NUMBER_REQUIRED_DA' in df.columns:
                    df_use = df[['BOM_NO', 'NO_BUMP', 'NUMBER_REQUIRED_DA']]
                    df_use = df_use.rename(columns={'NUMBER_REQUIRED_DA': 'NUMBER_REQUIRED'})
                else:
                    print(f"⚠️ ไม่พบคอลัมน์ NUMBER_REQUIRED หรือ NUMBER_REQUIRED_DA ในไฟล์ '{file}'")
                    skipped_files.append(file)
                    continue

                nobump_data.append(df_use)
                print(f"✅ โหลดข้อมูล NO_BUMP สำเร็จจาก '{file}'")
            else:
                print(f"⚠️ ไม่พบคอลัมน์ BOM_NO หรือ NO_BUMP ในไฟล์ '{file}'")
                skipped_files.append(file)

        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดกับไฟล์ '{file}': {e}")
            skipped_files.append(file)

    if nobump_data:
        combined_df = pd.concat(nobump_data, ignore_index=True)
        print(f"📊 รวมข้อมูล NO_BUMP สำเร็จทั้งหมด {len(combined_df)} แถว")
    else:
        combined_df = pd.DataFrame(columns=['BOM_NO', 'NO_BUMP', 'NUMBER_REQUIRED'])
        print("❌ ไม่พบข้อมูล NO_BUMP ที่สามารถใช้งานได้")

    if skipped_files:
        print(f"📌 ไฟล์ที่ข้ามเนื่องจากปัญหา: {skipped_files}")

    return combined_df

def load_packtype_auto():
    packtype_dir = os.path.join("data", "packtype")
    packtype_df = pd.DataFrame()

    try:
        files = glob.glob(os.path.join(packtype_dir, "*.xlsx"))
        if not files:
            print("⚠️ ไม่พบไฟล์ .xlsx ใน data/packtype/")
            return pd.DataFrame(columns=["bom_no", "assy_pack_type"])

        all_data = []
        for file in files:
            try:
                df = pd.read_excel(file)
                if 'bom_no' in df.columns and 'assy_pack_type' in df.columns:
                    df = df[['bom_no', 'assy_pack_type']].dropna()
                    df['bom_no'] = df['bom_no'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                    df['assy_pack_type'] = df['assy_pack_type'].astype(str).str.strip()
                    all_data.append(df)
                    print(f"✅ โหลดไฟล์: {os.path.basename(file)} — แถว: {len(df)}")
                else:
                    print(f"⚠️ ไฟล์ {os.path.basename(file)} ไม่มีคอลัมน์ 'bom_no' หรือ 'assy_pack_type'")
            except Exception as e:
                print(f"❌ โหลดไฟล์ {os.path.basename(file)} ล้มเหลว: {e}")

        if all_data:
            packtype_df = pd.concat(all_data, ignore_index=True).drop_duplicates()
            print(f"📦 รวมข้อมูลทั้งหมด: {len(packtype_df)} แถว จาก {len(files)} ไฟล์")
        else:
            print("⚠️ ไม่มีข้อมูลที่ตรงเงื่อนไขเลยในไฟล์ทั้งหมด")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

    return packtype_df

def load_packagecode_auto():
    packagecode_dir = os.path.join("data", "package code")
    packagecode_df = pd.DataFrame()

    try:
        files = glob.glob(os.path.join(packagecode_dir, "*.xlsx"))
        if not files:
            print("⚠️ ไม่พบไฟล์ .xlsx ใน data/package code/")
            return pd.DataFrame(columns=["bom_no", "package_code"])

        all_data = []
        for file in files:
            try:
                df = pd.read_excel(file)

                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

                if 'bom_no' in df.columns and 'package_code' in df.columns:
                    df = df[['bom_no', 'package_code']].dropna()
                    df['bom_no'] = df['bom_no'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.upper()
                    df['package_code'] = df['package_code'].astype(str).str.strip()
                    all_data.append(df)
                    print(f"✅ โหลดไฟล์: {os.path.basename(file)} — แถว: {len(df)}")
                else:
                    print(f"⚠️ ไฟล์ {os.path.basename(file)} ไม่มีคอลัมน์ 'bom_no' หรือ 'package_code'")
            except Exception as e:
                print(f"❌ โหลดไฟล์ {os.path.basename(file)} ล้มเหลว: {e}")

        if all_data:
            packagecode_df = pd.concat(all_data, ignore_index=True).drop_duplicates(subset="bom_no")
            print(f"📦 รวมข้อมูลทั้งหมด: {len(packagecode_df)} แถว จาก {len(all_data)} ไฟล์ที่โหลดได้")
        else:
            print("⚠️ ไม่มีข้อมูลที่ตรงเงื่อนไขเลยในไฟล์ทั้งหมด")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

    return packagecode_df

def cleanup_old_images(used_filenames, folder='static'):
    for fname in os.listdir(folder):
        if fname.endswith(('.png', '.jpg', '.jpeg')) and fname not in used_filenames:
            try:
                os.remove(os.path.join(folder, fname))
                print(f"🗑 ลบภาพ: {fname}")
            except Exception as e:
                print(f"❌ ลบภาพ {fname} ไม่สำเร็จ: {e}")

def get_used_filenames_somehow():
    return ['boxplot_after.png', 'boxplot_before.png']

def auto_cleanup():
    while True:
        used_filenames = get_used_filenames_somehow()
        cleanup_old_images(used_filenames, folder='static')
        time.sleep(60)

threading.Thread(target=auto_cleanup, daemon=True).start()

def process_all_files_in_data():
    all_data = []

    for fname in os.listdir(DATA_FOLDER):
        if fname.endswith(('.xlsx', '.csv')) and not fname.startswith('~$'):
            fpath = os.path.join(DATA_FOLDER, fname)

            try:
                if fname.endswith('.xlsx'):
                    df = pd.read_excel(fpath)
                else:
                    df = pd.read_csv(fpath)

                df.columns = df.columns.str.strip()

                if 'Package code' in df.columns and 'UPH' in df.columns:
                    df = df[df['Package code'].notna()]
                    df = df[df['Package code'] != '#N/A']
                    df['UPH'] = pd.to_numeric(df['UPH'], errors='coerce')
                    df = df[df['UPH'].notna()]

                    all_data.append(df[['Package code', 'UPH']])
            except Exception as e:
                print(f"❌ Error loading {fname}: {e}")

    if not all_data:
        return pd.DataFrame(columns=['Package code', 'UPH'])

    combined_df = pd.concat(all_data, ignore_index=True)

    grouped = combined_df.groupby('Package code')['UPH'].mean().reset_index()
    grouped.columns = ['Package code', 'UPH']
    grouped = grouped.sort_values(by='UPH', ascending=False)

    return grouped

@app.route("/uph_by_package_code")
def uph_by_package_code():
    result_df = process_all_files_in_data()
    return result_df.to_html(index=False, float_format="%.2f")

@app.route('/show_operations')
def show_operations():
    folder = 'Operation name'
    os.makedirs(folder, exist_ok=True)

    tables = {}

    for file in os.listdir(folder):
        if file.endswith('.csv'):
            try:
                path = os.path.join(folder, file)
                df = pd.read_csv(path)

                if 'operation' in df.columns:
                    # แปลงแนวตั้งเป็นแนวนอน
                    transposed = pd.DataFrame([df['operation'].dropna().tolist()])
                    tables[file] = transposed.to_html(index=False, header=True)
            except Exception as e:
                print(f"⚠️ อ่านไฟล์ {file} ไม่ได้: {e}")

    return render_template('show_operations.html', tables=tables if tables else None)

generated_files = []

@app.route("/url")
def url():
    return render_template("select_bom.html", files=generated_files)

@app.route("/notify_apl_done", methods=["POST"])
def notify_apl_done():
    data = request.get_json()
    filename = data.get("filename")
    if filename and filename not in generated_files:
        generated_files.append(filename)
        print(f"\u2705 ไฟล์ใหม่: {filename}")
    return jsonify({"status": "ok"})

@app.route("/check_new_files")
def check_new_files():
    return jsonify({"files": generated_files})

@app.route("/mock_add_file")
def mock_add_file():
    test_filename = "APL_test_file.xlsx"
    if test_filename not in generated_files:
        generated_files.append(test_filename)
        print(f"✅ Mock เพิ่มไฟล์: {test_filename}")
    return f"เพิ่มไฟล์ {test_filename} เรียบร้อย"

@app.route("/get_bom_list")
def get_bom_list():
    try:
        file_display_name = request.args.get('file')  # เช่น 'Data WB UTL1 Jan-May-25'
        if not file_display_name:
            return jsonify({"error": "Missing file name"}), 40

        print(f"📌 Requested File: {file_display_name}")

        file_map = {
            os.path.splitext(f)[0]: f 
            for f in os.listdir(DATA_FOLDER) 
            if f.lower().endswith(('.csv', '.xlsx')) and not f.startswith('~$')
}
        selected_file_exact = file_map.get(file_display_name)

        if not selected_file_exact:
            print("❌ File not found in mapping!")
            return jsonify({"error": f"File '{file_display_name}' not found"}), 404

        selected_file = selected_file_exact

        if not selected_file:
            return jsonify({"error": "File not found"}), 404

        file_path = os.path.join(DATA_FOLDER, selected_file)

        if selected_file.endswith('.csv'):
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(file_path)

        df.columns = df.columns.str.strip().str.lower()

        if 'bom_no' not in df.columns:
            print("❌ 'bom_no' column not found!")
            return jsonify({"error": "Missing 'bom_no' column in the file."}), 400

        df.dropna(subset=['bom_no'], inplace=True)
        bom_list = sorted(df['bom_no'].astype(str).unique())

        print(f"✅ BOM List Found: {bom_list}")
        return jsonify(bom_list)

    except Exception as e:
        print(f"❌ Exception Occurred: {e}")
        return jsonify({"error": str(e)}), 500
    
def get_latest_data_file(folder='data'):
    """คืนค่าชื่อไฟล์ล่าสุดในโฟลเดอร์ที่กำหนด"""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.csv', '.xlsx'))]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_user_selected_file(user_selected_name=None, file_type_filter=None):
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv")) + glob.glob(os.path.join(DATA_FOLDER, "*.xlsx"))

    if file_type_filter:
        all_files = [f for f in all_files if file_type_filter.upper() in os.path.basename(f).upper()]

    if user_selected_name:
        for f in all_files:
            if os.path.basename(f).strip().lower() == user_selected_name.strip().lower():
                return os.path.basename(f)
        print(f"❌ ไม่พบไฟล์ชื่อ: {user_selected_name}")
        return None

    print("⚠️ กรุณาเลือกไฟล์ข้อมูล")
    return None

def clean_filename_part(s):
    if not s:
        return ''
    return str(s).replace('\r', '').replace('\n', '').replace('_x000D_', '').strip()

def save_plots(df_before, df_after, data_type):
    plots_before = []
    plots_after = []

    df_before_type = df_before[df_before['data_type'] == data_type]
    df_after_type = df_after[df_after['data_type'] == data_type]

    model_col = None
    for col in df_before_type.columns:
        if col.strip().lower().replace(" ", "_") == 'machine_model':
            model_col = col
            break
    if not model_col:
        raise KeyError("ไม่พบคอลัมน์ Machine Model หรือ Machine_Model ในข้อมูล")    
    
    for model in df_before_type[model_col].unique():
        clean_model = clean_filename_part(model).replace('/', '_').replace(' ', '_')

        df_model_before = df_before_type[df_before_type[model_col] == model]
        if not df_model_before.empty:
            filename_before = f"boxplot_before_{data_type}_{clean_model}.png"
            plt.figure(figsize=(6, 4))
            df_model_before.boxplot(column='uph')
            plt.title(f"before  Outlier - {model}")
            plt.tight_layout()
            plt.savefig(os.path.join('static', filename_before))
            plt.close()
            plots_before.append(filename_before)

        df_model_after = df_after_type[df_after_type[model_col] == model]
        if not df_model_after.empty:
            filename_after = f"boxplot_after_{data_type}_{clean_model}.png"
            plt.figure(figsize=(6, 4))
            df_model_after.boxplot(column='uph')
            plt.title(f"after  Outlier - {model}")
            plt.tight_layout()
            plt.savefig(os.path.join('static', filename_after))
            plt.close()
            plots_after.append(filename_after)

    return plots_before, plots_after

def merge_data(before_data, after_data, data_type=None):
    if data_type:
        before_data = [d for d in before_data if d.get('Data_Type') == data_type]
        after_data = [d for d in after_data if d.get('Data_Type') == data_type]

    merged = []
    for before in before_data:
        model = before["Model"]
        after = next((a for a in after_data if a["Model"] == model), {})
        merged.append({
            **before,
            "Count_UPH_After": after.get("Count_UPH_After", 0),
            "Mean_After": after.get("Mean", 0),
            "STDEV_After": after.get("STDEV", 0),
            "Removed": after.get("Removed", 0),
            "Method": after.get("Method", "-"),
            "NO_BUMP": after.get("NO_BUMP", 0),
            "Wire_Per_Unit": after.get("Wire_Per_Unit", "-")
        })
    return merged

def truncate(text, length=20):
    if not isinstance(text, str):
        text = str(text)
    if len(text) > length:
        return text[:length] + "..."
    return text
def is_wb_file(filename):
    return 'wb' in filename.lower()

def is_pnp_file(filename):
    return 'pnp' in filename.lower()

def get_nobump_data(bom_no, file_type):
    for fname in os.listdir("temp"):
        if "wire" in fname.lower() and "data" in fname.lower():
            df = pd.read_excel(os.path.join("temp", fname))
            df.columns = df.columns.str.strip().str.upper()
            df['BOM_NO'] = df['BOM_NO'].astype(str)

            filtered = df[df['BOM_NO'] == bom_no]

            if file_type == 'WB':
                if 'NO_BUMP' in filtered.columns and 'NUMBER_REQUIRED' in filtered.columns:
                    return filtered[['NO_BUMP', 'NUMBER_REQUIRED']]
                else:
                    return pd.DataFrame({'NO_BUMP': [0], 'NUMBER_REQUIRED': [0]})
            else: 
                if 'NO_BUMP' in filtered.columns and 'NUMBER_REQUIRED_DA' in filtered.columns:
                    return filtered[['NO_BUMP', 'NUMBER_REQUIRED_DA']]
                else:
                    return pd.DataFrame({'NO_BUMP': [0], 'NUMBER_REQUIRED_DA': [1]})  # DA default = 1 unit

    if file_type == 'WB':
        return pd.DataFrame({'NO_BUMP': [0], 'NUMBER_REQUIRED': [0]})
    else:
        return pd.DataFrame({'NO_BUMP': [0], 'NUMBER_REQUIRED_DA': [1]})

@app.route("/display_data", methods=['GET', 'POST'])
def display_data():
    try:
        global csv_file_map

        def clean_text(s):
            if pd.isna(s):
                return s
            return str(s).replace('\r', '').replace('\n', '').replace('_x000D_', '').strip()

        if request.method == 'POST':
            selected_file = request.form.get('csv_file')
            selected_bom = request.form.get('bom')
            selected_operation = request.form.get('operation')
        else:
            selected_file = request.args.get('file')
            selected_bom = request.args.get('bom')
            selected_operation = request.args.get('operation')

        if not selected_file:
            latest_file = get_latest_data_file()
            if latest_file is None:
                return "ไม่พบไฟล์ข้อมูลในโฟลเดอร์ data"
            selected_file = os.path.basename(latest_file)

        if not selected_file.endswith(('.xlsx', '.csv')):
            if os.path.exists(os.path.join('data', selected_file + '.xlsx')):
                selected_file += '.xlsx'
            elif os.path.exists(os.path.join('data', selected_file + '.csv')):
                selected_file += '.csv'

        file_path = os.path.join('data', selected_file)
        if not os.path.exists(file_path):
            return f"ไม่พบไฟล์: {file_path}"

        if selected_file.endswith('.csv'):
            df_raw = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            df_raw = pd.read_excel(file_path)

        df_raw.columns = df_raw.columns.str.strip().str.lower()
        df_raw['uph'] = pd.to_numeric(df_raw['uph'], errors='coerce')
        df_raw.dropna(subset=['uph'], inplace=True)

        if 'bom_no' in df_raw.columns:
            df_raw['bom_no'] = df_raw['bom_no'].astype(str)

        model_col = None
        for col in df_raw.columns:
            if col.strip().lower().replace(" ", "_") == 'machine_model':
                model_col = col
                break
        if not model_col:
            return "ไม่พบคอลัมน์ Machine Model หรือ Machine_Model ในข้อมูล"

        for col in [model_col, 'bom_no', 'operation']:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].apply(clean_text)

        df_filtered = df_raw.copy()
        if selected_bom and 'bom_no' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['bom_no'] == selected_bom]
        if selected_operation and 'operation' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['operation'] == selected_operation]

        filename_lower = selected_file.lower()
        if 'wb' in filename_lower or 'lead' in filename_lower:
            data_type = 'wb'
        elif 'da' in filename_lower or 'die' in filename_lower:
            data_type = 'die'
        elif 'pnp' in filename_lower or 'pgk' in filename_lower:
            data_type = 'pnp'
        else:
            data_type = 'unknown'

        if data_type == 'pnp':
            if 'package_code' in df_filtered.columns:
                df_filtered['package_code'] = (
                    df_filtered['package_code'].astype(str)
                    .str.strip()
                    .replace({'#N/A': '', 'nan': ''})
                )
                pkg_counts = df_filtered[df_filtered['package_code'] != ''].groupby('package_code')['bom_no'].nunique()
                shared_pkg_codes = pkg_counts[pkg_counts > 1].index
                if not shared_pkg_codes.empty:
                    df_filtered = df_filtered[
                        df_filtered['package_code'].isin(shared_pkg_codes) |
                        (df_filtered['package_code'] == '')
                    ]

        def process(df_part, data_type, model_col):
            if df_part.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], pd.DataFrame()

            df_part['data_type'] = data_type

            count_before = df_part.groupby(model_col)['uph'].count().reset_index(name='Count Before')
            summary_before = df_part.groupby(model_col)['uph'].agg(['mean', 'std']).round(2).reset_index()
            summary_before = pd.merge(summary_before, count_before, on=model_col, how='left')

            df_clean = remove_outliers(df_part.copy())
            df_clean['data_type'] = data_type

            summary_after = df_clean.groupby(model_col)['uph'].agg(['mean', 'std']).round(2).reset_index()
            count_after = df_clean.groupby(model_col)['uph'].count().reset_index(name='Count After')
            summary_after = pd.merge(summary_after, count_after, on=model_col, how='left')
            summary_after = pd.merge(summary_after, count_before, on=model_col, how='left')
            summary_after['Removed Outliers'] = summary_after['Count Before'] - summary_after['Count After']
            summary_after.drop(columns=['Count Before'], inplace=True)

            if not df_clean.empty:
                if is_wb_file(selected_file):
                    file_type = 'WB'
                elif is_pnp_file(selected_file):
                    file_type = 'PNP'
                else:
                    file_type = 'DIE'

                bom_nobump_df = get_nobump_data(selected_bom, file_type)

                if file_type == 'WB':
                    no_bump_val = bom_nobump_df['NO_BUMP'].dropna().iloc[0] if not bom_nobump_df['NO_BUMP'].dropna().empty else 0
                    number_required_val = bom_nobump_df['NUMBER_REQUIRED'].dropna().iloc[0] if 'NUMBER_REQUIRED' in bom_nobump_df.columns and not bom_nobump_df['NUMBER_REQUIRED'].dropna().empty else 0
                    UNIT = (no_bump_val / 2) + number_required_val if (no_bump_val or number_required_val) else 0
                else:
                    UNIT = 1

                summary_after['Wire Per Unit'] = UNIT
                summary_after['UPH'] = summary_after['mean'].apply(lambda x: truncate(x / UNIT, 3) if UNIT else 0)

            # สรุปเฉพาะ PNP ตาม PACKAGE_CODE
            if data_type == 'pnp' and 'package_code' in df_clean.columns:
                package_summary = df_clean.groupby('package_code')['uph'].agg(['mean', 'std', 'count']).round(2).reset_index()
            else:
                package_summary = pd.DataFrame()

            boxplots_before, boxplots_after = save_plots(df_part, df_clean, data_type, model_col)
            return summary_before, summary_after, df_clean, boxplots_before, boxplots_after, package_summary

        def save_plots(df_before_type, df_after_type, data_type, model_col):
            plots_before = []
            plots_after = []

            for model in df_before_type[model_col].unique():
                clean_model = clean_filename_part(model).replace('/', '_').replace(' ', '_')

                df_model_before = df_before_type[df_before_type[model_col] == model]
                if not df_model_before.empty:
                    filename_before = f"boxplot_before_{data_type}_{clean_model}.png"
                    plt.figure(figsize=(6, 4))
                    df_model_before.boxplot(column='uph')
                    plt.title(f"Before Outlier - {model}")
                    plt.tight_layout()
                    plt.savefig(os.path.join('static', filename_before))
                    plt.close()
                    plots_before.append(filename_before)

                df_model_after = df_after_type[df_after_type[model_col] == model]
                if not df_model_after.empty:
                    filename_after = f"boxplot_after_{data_type}_{clean_model}.png"
                    plt.figure(figsize=(6, 4))
                    df_model_after.boxplot(column='uph')
                    plt.title(f"After Outlier - {model}")
                    plt.tight_layout()
                    plt.savefig(os.path.join('static', filename_after))
                    plt.close()
                    plots_after.append(filename_after)

            return plots_before, plots_after

        summary_before, summary_after, df_clean, boxplots_before, boxplots_after, package_summary = process(df_filtered, data_type, model_col)

        summary_before_html = summary_before.to_html(classes='table table-bordered', index=False) if not summary_before.empty else "ไม่มีข้อมูล"
        summary_after_html = summary_after.to_html(classes='table table-bordered', index=False) if not summary_after.empty else "ไม่มีข้อมูล"

        if not df_clean.empty:
            summary_cleaned = df_clean[['bom_no', model_col, 'Outlier_Method']].drop_duplicates() if 'Outlier_Method' in df_clean.columns else df_clean[['bom_no', model_col]].drop_duplicates()
            summary_cleaned_html = summary_cleaned.to_html(classes='table table-sm table-bordered', index=False)
        else:
            summary_cleaned_html = "ไม่มีข้อมูล"

        if data_type == 'pnp' and not package_summary.empty:
            package_summary_html = package_summary.to_html(classes='table table-sm table-bordered', index=False)
        else:
            package_summary_html = "ไม่มีข้อมูล"

        return render_template("bom_detail.html",
                               selected_file=selected_file,
                               selected_bom=selected_bom,
                               selected_operation=selected_operation,
                               data_type=data_type,
                               summary_before=summary_before_html,
                               summary_after=summary_after_html,
                               data=summary_cleaned_html,
                               efficiency_table=package_summary_html,
                               boxplots_before=boxplots_before,
                               boxplots_after=boxplots_after)

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"
    
@app.route('/js/<path:filename>')
def serve_js(filename):
    print(f"📦 JS requested: {filename}")
    return send_from_directory(os.path.join(app.root_path, 'js'), filename)

@app.route("/main", methods=["GET"])
def main():
    plants = ["utl1", "utl2", "utl3"]
    year_quarters = ["2024Q1", "2024Q2", "2024Q3", "2025Q1", "2025Q2", "2025Q3"]
    operations = ["LEAD BOND ROV", "DIE ATTACH", "DIE ATTACH MAP", "PKG PICK PLACE"]

    print("✅ PLANTS:", plants)
    print("✅ QUARTERS:", year_quarters)
    print("✅ OPERATIONS:", operations)

    return render_template(
        "main.html",
        plants=plants,
        year_quarters=year_quarters,
        operations=operations
    )

@app.route("/download_apl_excel")
def download_apl_excel():
    plant = request.args.get("plant")
    year_quarter = request.args.get("year_quarter")
    operation = request.args.get("operation")

    if not all([plant, year_quarter, operation]):
        return "❌ Missing parameters", 400

    # 👇 ถ้า operation เป็น "LEAD BOND ROV_WB" ให้ใช้แค่ "LEAD BOND ROV" เรียก API
    actual_operation = operation.split("_WB")[0] if operation.endswith("_WB") else operation

    encoded_operation = urllib.parse.quote(actual_operation)
    base_api_url = (
        f"http://th3sroeeeng4/RTMSAPI/ApiAutoUph/api/data"
        f"?plant={plant}&year_quarter={year_quarter}&operation={encoded_operation}"
    )

    success, filepath = run_apl(base_api_url, plant, year_quarter, operation)  # ส่ง operation เดิมไว้ใช้ตั้งชื่อไฟล์

    if success:
        return send_file(filepath, as_attachment=True)
    else:
        return "❌ ไม่สามารถโหลดข้อมูลได้", 500

def run_apl(full_url, plant, year_quarter, operation):
    try:
        print(f"🌐 Fetching: {full_url}")
        response = requests.get(full_url, headers={"Accept": "application/json"})

        print(f"🔎 Status: {response.status_code}")
        print(f"🧾 Raw Response Text (first 300 chars): {response.text[:300]}")

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as json_err:
                print(f"❌ JSON decode error: {json_err}")
                return False, None

            if not isinstance(data, list):
                print(f"⚠️ Unexpected data format: {type(data)}")
                return False, None

            df_new = pd.DataFrame(data)

            print(f"✅ Loaded {len(df_new)} rows")

            # ตรวจว่าคอลัมน์ครบไหม
            required_cols = ['date_time_start', 'bom_no', 'operation', 'optn_code', 'Machine_Model', 'UPH']
            missing_cols = [col for col in required_cols if col not in df_new.columns]
            if missing_cols:
                print(f"❌ Missing columns in data: {missing_cols}")
                return False, None

            df_new = df_new[required_cols]

            os.makedirs("data", exist_ok=True)
            filename = f"APL_{plant}_{year_quarter}_{operation.replace(' ', '_')}.xlsx"
            save_path = os.path.join("data", filename)

            if os.path.exists(save_path):
                df_old = pd.read_excel(save_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(
                    subset=['date_time_start', 'bom_no', 'operation', 'optn_code', 'Machine_Model'],
                    keep='last'
                )
                df_combined.to_excel(save_path, index=False)
                print(f"✅ Updated and saved: {save_path}")
            else:
                df_new.to_excel(save_path, index=False)
                print(f"✅ Saved new APL to: {save_path}")

            return True, save_path
        else:
            print(f"❌ API Error: {response.status_code}")
            return False, None

    except Exception as e:
        print(f"❌ Exception during API call: {e}")
        return False, None
    
@app.route("/", methods=['GET', 'POST'])
def select_bom():
    all_files = [f for f in os.listdir(DATA_FOLDER)
                 if f.endswith(('.csv', '.xlsx')) and not f.startswith('~$')]

    wb_files, da_files, pnp_files = [], [], []

    for file in all_files:
        path = os.path.join(DATA_FOLDER, file)
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(path, nrows=10)
            else:
                df = pd.read_excel(path, nrows=10)

            content = ' '.join(df.astype(str).fillna('').values.ravel()).upper()

            if any(k in content for k in ['WB', 'LEAD']):
                wb_files.append(file)
            if any(k in content for k in ['PNP', 'PKG']):
                pnp_files.append(file)
            if any(k in content for k in ['DA', 'DIE']):
                da_files.append(file)

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    # รวมไฟล์ทั้งหมดที่อยู่ในกลุ่มต่าง ๆ ไว้ใน other_files
    other_files = list(set(wb_files + da_files + pnp_files))

    # เตรียมชื่อแสดงผล
    wb_display_names = [os.path.splitext(f)[0] for f in wb_files]
    da_display_names = [os.path.splitext(f)[0] for f in da_files]
    pnp_display_names = [os.path.splitext(f)[0] for f in pnp_files]
    other_display_names = [os.path.splitext(f)[0] for f in other_files]

    # แมปชื่อที่แสดงกับชื่อไฟล์จริง
    csv_file_map = {
        **{name: f for name, f in zip(wb_display_names, wb_files)},
        **{name: f for name, f in zip(da_display_names, da_files)},
        **{name: f for name, f in zip(pnp_display_names, pnp_files)},
        **{name: f for name, f in zip(other_display_names, other_files)},
    }

    selected_file = None
    selected_display_name = None
    selected_bom = None
    selected_operation = None
    selected_optn_code = None
    selected_plant = None
    selected_year_quarter = None
    selected_panel = None
    bom_list, operation_list, optn_code_list = [], [], []

    def read_bom_list(filename):
        full_path = os.path.join(DATA_FOLDER, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(full_path)
        else:
            df = pd.read_excel(full_path)
        df.columns = df.columns.str.strip().str.lower()
        bom = df['bom_no'].dropna().unique().tolist() if 'bom_no' in df.columns else []
        operation = df['operation'].dropna().unique().tolist() if 'operation' in df.columns else []
        optn_code = df['optn_code'].dropna().unique().tolist() if 'optn_code' in df.columns else []
        return bom, operation, optn_code

    if request.method == 'POST':
        selected_plant = request.form.get('plant')
        selected_year_quarter = request.form.get('year_quarter')

        if 'submit_wb' in request.form:
            selected_panel = 'wb'
            selected_display_name = request.form.get('wb_csv_file')
            selected_file = csv_file_map.get(selected_display_name)
            selected_bom = request.form.get('wb_selected_bom')
            selected_operation = request.form.get('wb_operation')
            selected_optn_code = request.form.get('wb_optn_code')

        elif 'submit_da' in request.form:
            selected_panel = 'da'
            selected_display_name = request.form.get('da_csv_file')
            selected_file = csv_file_map.get(selected_display_name)
            selected_bom = request.form.get('da_selected_bom')
            selected_operation = request.form.get('da_operation')
            selected_optn_code = request.form.get('da_optn_code')

        elif 'submit_pnp' in request.form:
            selected_panel = 'pnp'
            selected_display_name = request.form.get('pnp_csv_file')
            selected_file = csv_file_map.get(selected_display_name)
            selected_bom = request.form.get('pnp_selected_bom')
            selected_operation = request.form.get('pnp_operation')
            selected_optn_code = request.form.get('pnp_optn_code')

        elif 'submit_other' in request.form:
            selected_panel = 'other'
            selected_display_name = request.form.get('other_csv_file')
            selected_file = csv_file_map.get(selected_display_name)
            selected_bom = request.form.get('other_selected_bom')
            selected_operation = request.form.get('other_operation')
            selected_optn_code = request.form.get('other_optn_code')

        if selected_file:
            bom_list, operation_list, optn_code_list = read_bom_list(selected_file)

            if selected_plant and selected_year_quarter and selected_bom and selected_operation:
                safe_operation = selected_operation.replace(" ", "%20")
                base_url = f"http://th3sroeeeng4/RTMSAPI/ApiAutoUph/api/data?plant={selected_plant}&year_quarter={selected_year_quarter}&operation={safe_operation}"

                try:
                    run_apl(base_url, selected_bom)
                    flash("✅ APL fetched and saved successfully!", "success")
                except Exception as e:
                    flash(f"❌ Failed to fetch APL: {e}", "danger")

    return render_template(
        "select_bom.html",
        wb_files=wb_display_names,
        da_files=da_display_names,
        pnp_files=pnp_display_names,
        other_files=other_display_names,

        wb_bom_list=bom_list if selected_panel == 'wb' else [],
        da_bom_list=bom_list if selected_panel == 'da' else [],
        pnp_bom_list=bom_list if selected_panel == 'pnp' else [],
        other_bom_list=bom_list if selected_panel == 'other' else [],

        wb_operation_list=operation_list if selected_panel == 'wb' else [],
        da_operation_list=operation_list if selected_panel == 'da' else [],
        pnp_operation_list=operation_list if selected_panel == 'pnp' else [],
        other_operation_list=operation_list if selected_panel == 'other' else [],

        wb_optn_code_list=optn_code_list if selected_panel == 'wb' else [],
        da_optn_code_list=optn_code_list if selected_panel == 'da' else [],
        pnp_optn_code_list=optn_code_list if selected_panel == 'pnp' else [],
        other_optn_code_list=optn_code_list if selected_panel == 'other' else [],

        selected_wb_file=selected_display_name if selected_panel == 'wb' else None,
        selected_da_file=selected_display_name if selected_panel == 'da' else None,
        selected_pnp_file=selected_display_name if selected_panel == 'pnp' else None,
        selected_other_file=selected_display_name if selected_panel == 'other' else None,

        selected_wb_bom=selected_bom if selected_panel == 'wb' else None,
        selected_da_bom=selected_bom if selected_panel == 'da' else None,
        selected_pnp_bom=selected_bom if selected_panel == 'pnp' else None,
        selected_other_bom=selected_bom if selected_panel == 'other' else None,

        selected_wb_operation=selected_operation if selected_panel == 'wb' else None,
        selected_da_operation=selected_operation if selected_panel == 'da' else None,
        selected_pnp_operation=selected_operation if selected_panel == 'pnp' else None,
        selected_other_operation=selected_operation if selected_panel == 'other' else None,

        selected_wb_optn_code=selected_optn_code if selected_panel == 'wb' else None,
        selected_da_optn_code=selected_optn_code if selected_panel == 'da' else None,
        selected_pnp_optn_code=selected_optn_code if selected_panel == 'pnp' else None,
        selected_other_optn_code=selected_optn_code if selected_panel == 'other' else None,

        selected_plant=selected_plant,
        selected_year_quarter=selected_year_quarter,
    )

def get_csv_file_map():
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv') or f.endswith('.xlsx')]
    # สมมติชื่อแสดงคือชื่อไฟล์เหมือนกัน
    csv_file_map = {f: f for f in files}
    return csv_file_map

def truncate(number, decimals=3):
    if not isinstance(number, (int, float)):
        return number
    factor = 10 ** decimals
    return int(number * factor) / factor

def clean_text(s):
    if isinstance(s, str):
        return s.strip().replace('\r', '').replace('\n', '')
    return s

@app.route("/all_boms", methods=['GET', 'POST'])
def all_boms():
    nobump_df = load_all_nobump_data()

    csv_files = sorted([
        f for f in os.listdir(DATA_FOLDER)
        if f.endswith(('.csv', '.xlsx')) and not f.startswith('~$')
    ])

    selected_file = None
    summary_wb = []
    summary_da = []
    summary_pnp = []

    def is_wb_file(filename):
        return 'wb' in filename.lower() or 'wire' in filename.lower()

    def is_pnp_file(filename):
        return 'pnp' in filename.lower()

    def clean_text(s):
        if pd.isna(s):
            return s
        return str(s).replace('\r', '').replace('\n', '').replace('_x000D_', '').strip()

    packtype_df = pd.DataFrame()
    try:
        dfs = []
        for f in os.listdir('packtype'):
            if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$'):
                df = pd.read_excel(os.path.join('packtype', f))
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                if 'bom_no' in df.columns and 'assy_pack_type' in df.columns:
                    df = df[['bom_no', 'assy_pack_type']].dropna()
                    df['bom_no'] = df['bom_no'].astype(str).str.strip().str.upper()
                    df['assy_pack_type'] = df['assy_pack_type'].astype(str).str.strip()
                    dfs.append(df)
        if dfs:
            packtype_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='bom_no')
    except Exception as e:
        print(f"❌ โหลด packtype ไม่สำเร็จ: {e}")

    packagecode_df = pd.DataFrame()
    try:
        dfs = []
        for f in os.listdir('package code'):
            if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$'):
                df = pd.read_excel(os.path.join('package code', f))
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                if 'bom_no' in df.columns and 'package_code' in df.columns:
                    df = df[['bom_no', 'package_code']].dropna()
                    df['bom_no'] = df['bom_no'].astype(str).str.strip().str.upper()
                    df['package_code'] = df['package_code'].astype(str).str.strip()
                    dfs.append(df)
        if dfs:
            packagecode_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='bom_no')
    except Exception as e:
        print(f"❌ โหลด package_code ไม่สำเร็จ: {e}")

    if request.method == 'POST':
        selected_file = request.form.get('csv_file')
        if selected_file and selected_file in csv_files:
            file_path = os.path.join(DATA_FOLDER, selected_file)
            try:
                if selected_file.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding="utf-8-sig")
                else:
                    df = pd.read_excel(file_path)

                df.columns = [col.strip() for col in df.columns]

                model_col = next((col for col in df.columns if col.lower().replace("_", " ") == "machine model"), None)
                if not model_col:
                    return "ไม่พบคอลัมน์ Machine Model หรือ Machine_Model ในไฟล์"

                df['UPH'] = pd.to_numeric(df['UPH'], errors='coerce')
                df[model_col] = df[model_col].apply(clean_text)

                if {'bom_no', 'Package code', model_col, 'UPH'}.issubset(df.columns):
                    df.rename(columns={'Package code': 'package_code', model_col: 'model', 'UPH': 'uph'}, inplace=True)
                    df['model'] = df['model'].apply(clean_text)
                    df['package_code'] = df['package_code'].astype(str).str.strip()
                    df['uph'] = pd.to_numeric(df['uph'], errors='coerce')
                    df.dropna(subset=['package_code', 'model', 'uph'], inplace=True)

                    grouped = df.groupby(['package_code', 'model'])['uph'].agg(['mean', 'std', 'count']).round(2).reset_index()

                    for _, row in grouped.iterrows():
                        assy_pack_val = "TUBE" if "NX-116" in row["model"].upper() else ""
                        if not assy_pack_val:
                            matched_bom = df[df['model'] == row['model']]['bom_no'].dropna().unique()
                            for bom_candidate in matched_bom:
                                match_pack = packtype_df[packtype_df['bom_no'] == bom_candidate]['assy_pack_type']
                                if not match_pack.empty:
                                    assy_pack_val = match_pack.iloc[0]
                                    break
                            if not assy_pack_val:
                                assy_pack_val = "ไม่พบ packtype"

                        summary_pnp.append({
                            "bom": "",
                            "model": row["model"],
                            "mean_after": row["mean"],
                            "adjusted_mean": 1,
                            "wire_per_unit": 1,
                            "efficiency_ratio": truncate(row["mean"], 3),
                            "no_outlier_removed": "",
                            "operation": "",
                            "optn_code": "",
                            "pnp_specific": "รวมตาม Package code (แม้มี BOM ก็ไม่ใช้)",
                            "assy_pack_type": assy_pack_val,
                            "package_code": row["package_code"]
                        })

                    selected_file_display = os.path.splitext(selected_file)[0] if selected_file else None
                    return render_template(
                        "all_boms.html",
                        summary_wb=[],
                        summary_da=[],
                        summary_pnp=summary_pnp,
                        package_summary=grouped.to_dict(orient='records'),
                        package_aggregated=grouped.to_dict(orient='records'),
                        file_list=csv_files,
                        selected_file=selected_file,
                        selected_file_display=selected_file_display
                    )

                # ✅ กรณีปกติ (มี bom_no)
                df['bom_no'] = df['bom_no'].astype(str).apply(clean_text).str.upper()
                df.dropna(subset=['UPH', model_col, 'bom_no'], inplace=True)

                for bom in sorted(df['bom_no'].unique()):
                    df_bom = df[df['bom_no'] == bom]
                    df_clean = remove_outliers(df_bom)

                    operation_val = df_bom['operation'].dropna().iloc[0] if 'operation' in df_bom.columns and not df_bom['operation'].dropna().empty else ""
                    optn_code_val = df_bom['optn_code'].dropna().iloc[0] if 'optn_code' in df_bom.columns and not df_bom['optn_code'].dropna().empty else ""

                    df_clean['Normalized Model'] = df_clean[model_col].apply(
                        lambda x: 'WB3100' if 'WB3100' in x else x
                    )

                    summary_after = df_clean.groupby("Normalized Model")["UPH"].agg(['mean']).round(2)
                    bom_nobump_df = nobump_df[nobump_df['BOM_NO'] == bom]

                    if is_wb_file(selected_file):
                        file_type = 'WB'
                    elif is_pnp_file(selected_file):
                        file_type = 'PNP'
                    else:
                        file_type = 'DA'

                    if file_type == 'WB':
                        no_bump_val = bom_nobump_df['NO_BUMP'].dropna().iloc[0] if not bom_nobump_df['NO_BUMP'].dropna().empty else 0
                        number_required_val = (
                            bom_nobump_df['NUMBER_REQUIRED'].dropna().iloc[0]
                            if 'NUMBER_REQUIRED' in bom_nobump_df.columns and not bom_nobump_df['NUMBER_REQUIRED'].dropna().empty
                            else 0
                        )
                        UNIT = adjusted_val = (no_bump_val / 2) + number_required_val if (no_bump_val or number_required_val) else 1
                    else:
                        UNIT = 1

                    assy_pack_val = ""
                    package_code_val = ""
                    if file_type == "PNP":
                        if any(df_clean['Normalized Model'].str.upper().str.contains("NX-116")):
                            assy_pack_val = "TUBE"
                        else:
                            match_pack = packtype_df[packtype_df['bom_no'] == bom]['assy_pack_type']
                            assy_pack_val = match_pack.iloc[0] if not match_pack.empty else "ไม่พบ packtype"

                        match_code = packagecode_df[packagecode_df['bom_no'] == bom]['package_code']
                        package_code_val = match_code.iloc[0] if not match_code.empty else "ไม่พบ package_code"

                    for model in df_clean['Normalized Model'].unique():
                        df_clean_model = df_clean[df_clean['Normalized Model'] == model]
                        uph_mean = summary_after.loc[model]['mean'] if model in summary_after.index else 0

                        no_outlier_removed = (
                            'Outlier_Method' in df_clean_model.columns and
                            df_clean_model['Outlier_Method'].iloc[0] == 'ไม่ตัด (ข้อมูลน้อย)'
                        )
                        outlier_note = f"ไม่ตัด Outlier (ข้อมูลน้อย) — แถว: {len(df_clean_model)}" if no_outlier_removed else ""

                        eff_ratio = truncate(uph_mean / UNIT, 3) if UNIT else 0

                        row = {
                            "bom": bom,
                            "model": model,
                            "mean_after": uph_mean,
                            "adjusted_mean": round(UNIT, 2),
                            "wire_per_unit": round(UNIT, 2),
                            "efficiency_ratio": eff_ratio,
                            "no_outlier_removed": outlier_note,
                            "operation": operation_val,
                            "optn_code": optn_code_val
                        }

                        if file_type == 'WB':
                            row["wb_specific"] = "ข้อมูลเฉพาะ WB"
                            summary_wb.append(row)
                        elif file_type == 'PNP':
                            row["pnp_specific"] = "ข้อมูลเฉพาะ PNP"
                            row["assy_pack_type"] = assy_pack_val
                            row["package_code"] = package_code_val
                            summary_pnp.append(row)
                        else:
                            row["da_specific"] = "ข้อมูลเฉพาะ DA"
                            summary_da.append(row)

            except Exception as e:
                print(f"❌ Error processing file {selected_file}: {e}")

    selected_file_display = os.path.splitext(selected_file)[0] if selected_file else None

    package_summary = pd.DataFrame()
    if summary_pnp:
        df_pnp = pd.DataFrame(summary_pnp)
        if 'package_code' in df_pnp.columns:
            df_valid = df_pnp[df_pnp['package_code'].notna() & (df_pnp['package_code'] != '')]
            package_summary = (
                df_valid.groupby('package_code')['efficiency_ratio']
                .agg(['mean', 'std', 'count'])
                .round(2)
                .reset_index()
            )

    package_aggregated = pd.DataFrame()
    if summary_pnp:
        df_pnp_all = pd.DataFrame(summary_pnp)
        df_pnp_valid = df_pnp_all[
            df_pnp_all['package_code'].notna() &
            (df_pnp_all['package_code'] != '') &
            df_pnp_all['efficiency_ratio'].notna()
        ]
        package_aggregated = (
            df_pnp_valid.groupby('package_code')['efficiency_ratio']
            .agg(['mean', 'std', 'count'])
            .round(2)
            .reset_index()
        )

    return render_template(
        "all_boms.html",
        summary_wb=summary_wb,
        summary_da=summary_da,
        summary_pnp=summary_pnp,
        package_summary=package_summary.to_dict(orient='records'),
        package_aggregated=package_aggregated.to_dict(orient='records'),
        file_list=csv_files,
        selected_file=selected_file,
        selected_file_display=selected_file_display
    )

def clean_text(s):
    if isinstance(s, str):
        return s.strip().replace('\r', '').replace('\n', '')
    return s

@app.route("/export_all_boms_excel", methods=['POST'])
def export_all_boms_excel():
    try:
        def clean_text(s):
            if pd.isna(s):
                return s
            return str(s).replace('\r', '').replace('\n', '').replace('_x000D_', '').strip()

        def safe_first(series):
            return series.dropna().iloc[0] if not series.dropna().empty else ""

        selected_file = request.form.get('csv_file')
        if not selected_file:
            return "No file selected", 400

        file_path = os.path.join(DATA_FOLDER, selected_file)
        if selected_file.endswith('.csv'):
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(file_path)

        df.columns = [c.strip() for c in df.columns]
        model_col = next((c for c in df.columns if c.lower().replace("_", " ") == "machine model"), None)
        if not model_col:
            return "ไม่พบคอลัมน์ Machine Model", 400
        df[model_col] = df[model_col].apply(clean_text)

        has_bom = 'bom_no' in df.columns
        has_pkg = 'Package code' in df.columns
        has_uph = 'UPH' in df.columns
        if not has_uph:
            return "ไม่พบคอลัมน์ UPH", 400
        df['UPH'] = pd.to_numeric(df['UPH'], errors='coerce')

        file_type = 'WB' if 'wb' in selected_file.lower() else 'PNP' if 'pnp' in selected_file.lower() else 'DA'

        summary_wb, summary_da, summary_pnp = [], [], []

        if has_bom:
            df['bom_no'] = df['bom_no'].astype(str).apply(clean_text).str.upper()

        if has_pkg:
            df['package_code'] = df['Package code'].astype(str).str.strip()
            df['Normalized Model'] = df[model_col].apply(clean_text)
            df.dropna(subset=['package_code', 'Normalized Model', 'UPH'], inplace=True)

            grouped = df.groupby(['package_code', 'Normalized Model'])['UPH'].agg(['mean']).reset_index()
            for _, row in grouped.iterrows():
                assy_pack_val = "TUBE" if "NX-116" in row['Normalized Model'].upper() else "ไม่พบ packtype"
                summary_pnp.append({
                    "bom": "",
                    "model": row['Normalized Model'],
                    "optn_code": "",
                    "operation": "",
                    "Wire Per Hour": round(row['mean'], 2),
                    "wire_per_unit": 1,
                    "UPH": round(row['mean'], 3),
                    "no_outlier_removed": "",
                    "assy_pack_type": assy_pack_val,
                    "package_code": row['package_code']
                })
        elif has_bom:
            nobump_df = load_all_nobump_data()
            packtype_df, package_code_df = pd.DataFrame(), pd.DataFrame()

            try:
                packtype_df = pd.concat([
                    pd.read_excel(os.path.join('packtype', f))[['bom_no', 'assy_pack_type']]
                    .dropna().assign(
                        bom_no=lambda x: x['bom_no'].astype(str).str.strip().str.upper(),
                        assy_pack_type=lambda x: x['assy_pack_type'].astype(str).str.strip()
                    )
                    for f in os.listdir('packtype')
                    if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')
                ], ignore_index=True).drop_duplicates(subset='bom_no')
            except: pass

            try:
                package_code_df = pd.concat([
                    pd.read_excel(os.path.join('package code', f))[['bom_no', 'package_code']]
                    .dropna().assign(
                        bom_no=lambda x: x['bom_no'].astype(str).str.strip().str.upper(),
                        package_code=lambda x: x['package_code'].astype(str).str.strip()
                    )
                    for f in os.listdir('package code')
                    if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')
                ], ignore_index=True).drop_duplicates(subset='bom_no')
            except: pass

            df.dropna(subset=['UPH', model_col, 'bom_no'], inplace=True)

            for bom in sorted(df['bom_no'].unique()):
                df_bom = df[df['bom_no'] == bom]
                df_clean = remove_outliers(df_bom)
                df_clean['Normalized Model'] = df_clean[model_col].apply(lambda x: 'WB3100' if 'WB3100' in x else x)
                summary = df_clean.groupby('Normalized Model')['UPH'].agg(['mean']).reset_index()

                count_before = len(df_bom)
                count_after = len(df_clean)
                if count_before < 15:
                    no_outlier_removed = f"ไม่ตัด Outlier (ข้อมูลน้อย) — แถว: {count_before}"
                else:
                    no_outlier_removed = f"ตัด Outlier — ก่อน: {count_before} หลัง: {count_after}"

                no_bump_val = safe_first(nobump_df[nobump_df['BOM_NO'] == bom]['NO_BUMP']) if file_type == 'WB' else 0
                number_required_val = safe_first(nobump_df[nobump_df['BOM_NO'] == bom]['NUMBER_REQUIRED']) if file_type == 'WB' else 0

                try:
                    no_bump_val = float(no_bump_val) if no_bump_val else 0
                    number_required_val = float(number_required_val) if number_required_val else 0
                except:
                    no_bump_val, number_required_val = 0, 0

                UNIT = adjusted_val = (no_bump_val / 2) + number_required_val if (no_bump_val or number_required_val) else 1

                for _, row in summary.iterrows():
                    model = row['Normalized Model']
                    mean_uph = row['mean']
                    eff_ratio = round(mean_uph / UNIT, 3)

                    result = {
                        "bom": bom,
                        "model": model,
                        "optn_code": safe_first(df_bom['optn_code']) if 'optn_code' in df_bom else "",
                        "operation": safe_first(df_bom['operation']) if 'operation' in df_bom else "",
                        "Wire Per Hour": round(mean_uph, 2),
                        "wire_per_unit": round(UNIT, 2),
                        "UPH": eff_ratio,
                        "no_outlier_removed": no_outlier_removed
                    }

                    if file_type == 'WB':
                        summary_wb.append(result)
                    elif file_type == 'PNP':
                        result["assy_pack_type"] = safe_first(packtype_df[packtype_df['bom_no'] == bom]['assy_pack_type']) if bom in packtype_df['bom_no'].values else "ไม่พบ packtype"
                        result["package_code"] = safe_first(package_code_df[package_code_df['bom_no'] == bom]['package_code']) if bom in package_code_df['bom_no'].values else "ไม่พบ package_code"
                        summary_pnp.append(result)
                    else:
                        summary_da.append(result)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if summary_wb:
                pd.DataFrame(summary_wb).to_excel(writer, sheet_name='WB Summary', index=False)
            if summary_da:
                pd.DataFrame(summary_da).to_excel(writer, sheet_name='DA Summary', index=False)
            if summary_pnp:
                pd.DataFrame(summary_pnp).to_excel(writer, sheet_name='PNP Summary', index=False)

        output.seek(0)
        return send_file(output, download_name=f"{os.path.splitext(selected_file)[0]}_summary.xlsx", as_attachment=True)

    except Exception as e:
        return f"Error exporting summary: {e}", 500
#
def standardize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

@app.route('/frame_stock', methods=['GET', 'POST'])
def frame_stock():
    global processed_df
    files = [f for f in os.listdir(FRAMESTOCK_FOLDER) if f.endswith('.xlsx') and not f.startswith('~$')]

    selected_file = None
    data = []

    package_folder = 'package and frame stock'
    map_file = next((f for f in os.listdir(package_folder) if f.endswith('.xlsx')), None)
    item_map = {}

    if map_file:
        map_path = os.path.join(package_folder, map_file)
        map_df = pd.read_excel(map_path)
        if 'ITEM_NO' in map_df.columns and 'PACKAGE_CODE' in map_df.columns:
            item_map = dict(zip(map_df['ITEM_NO'].astype(str).str.strip(), map_df['PACKAGE_CODE'].astype(str).str.strip()))

    if request.method == 'POST':
        selected_file = request.form.get('file')
        if selected_file:
            filepath = os.path.join(FRAMESTOCK_FOLDER, selected_file)
            df = pd.read_excel(filepath, skiprows=2)
            df = standardize_columns(df)

            op_col = 'Unnamed: 2'
            time_col = 'Unnamed: 1'
            step_col = 'Unnamed: 5'
            machine_col = 'Unnamed: 3'
            st_col = 'Unnamed: 3'
            speed_col = 'Unnamed: 10'

            target_ops = ['PRO', 'CUC', 'ERRSET', 'ERRRCV', 'ERRCLR', 'DMC', 'DMW']
            df = df[df[op_col].astype(str).str.upper().isin(target_ops)].reset_index(drop=True)

            df[step_col] = pd.to_numeric(df[step_col], errors='coerce')
            df[speed_col] = pd.to_numeric(df.get(speed_col, pd.Series(dtype=float)), errors='coerce')
            df['Speed'] = (df[speed_col] / 10 / 25.4).round().astype('Int64')
            df['SPEED'] = None
            df['sec'] = None
            df['min'] = None
            df['Reverse'] = None
            df['Group'] = None
            df['Average'] = ''
            df['Data Point'] = ''
            df['DateOnly'] = pd.to_datetime(df['Unnamed: 0'], errors='coerce').dt.date

            for i, row in df.iterrows():
                if row[op_col] == 'PRO':
                    machine_id = row[machine_col]
                    cuc_rows = df.loc[:i - 1]
                    matched_cuc = cuc_rows[
                        (cuc_rows[op_col] == 'CUC') &
                        (cuc_rows[machine_col] == machine_id) &
                        (cuc_rows['Speed'].notna())
                    ]
                    if not matched_cuc.empty:
                        df.at[i, 'SPEED'] = matched_cuc.iloc[-1]['Speed']

            pro_df = df[df[op_col] == 'PRO']
            pro_indices = pro_df.index.tolist()

            groups = []
            current_group = []
            last_machine = None

            for idx in pro_indices:
                step = df.at[idx, step_col]
                machine = df.at[idx, st_col]

                if pd.isna(step):
                    continue

                if last_machine is not None and machine != last_machine:
                    if len(current_group) > 1:
                        groups.append(current_group.copy())
                    current_group = []

                current_group.append(idx)
                last_machine = machine

                if step == 1:
                    if len(current_group) > 1:
                        groups.append(current_group.copy())
                    current_group = []
                    last_machine = None

            if len(current_group) > 1:
                groups.append(current_group.copy())

            error_ops = ['ERRSET', 'ERRRCV', 'ERRCLR', 'DMC', 'DMW']
            all_deltas = []
            group_date_tracker = {}

            for group in groups:
                if len(group) < 2:
                    continue

                group_df = df.loc[group].copy()
                group_df['DateOnly'] = pd.to_datetime(group_df['Unnamed: 0'], errors='coerce').dt.date
                date_groups = group_df.groupby('DateOnly')

                for date, sub_df in date_groups:
                    sub_group = sub_df.index.tolist()
                    if len(sub_group) < 2:
                        continue

                    date_str = str(date) if pd.notnull(date) else 'Unknown'
                    group_date_tracker.setdefault(date_str, 0)
                    group_date_tracker[date_str] += 1
                    group_number = group_date_tracker[date_str]
                    group_name = f"Group {group_number} ({date_str})"

                    last_step = df.at[sub_group[-1], step_col]
                    if last_step != 1:
                        for idx in sub_group:
                            df.at[idx, 'Group'] = f"Invalid {group_name}"
                        continue

                    deltas = []
                    for idx in sub_group:
                        df.at[idx, 'Group'] = group_name

                    for i in range(1, len(sub_group)):
                        prev_idx = sub_group[i - 1]
                        curr_idx = sub_group[i]

                        step_prev = df.at[prev_idx, step_col]
                        step_curr = df.at[curr_idx, step_col]

                        if pd.notnull(step_prev) and pd.notnull(step_curr) and step_curr > step_prev:
                            df.at[curr_idx, 'sec'] = 'Out of Order'
                            df.at[curr_idx, 'min'] = 'Out of Order'
                            df.at[curr_idx, 'Reverse'] = 'Out of Order'
                            continue

                        between = df.loc[prev_idx + 1: curr_idx - 1]
                        has_error = between[op_col].astype(str).str.upper().isin(error_ops).any()

                        if has_error:
                            df.at[curr_idx, 'sec'] = 'Machine Error'
                            df.at[curr_idx, 'min'] = 'Machine Error'
                            df.at[curr_idx, 'Reverse'] = 'Machine Error'
                            continue

                        t1 = pd.to_datetime(df.at[prev_idx, time_col], format='%H:%M:%S', errors='coerce')
                        t2 = pd.to_datetime(df.at[curr_idx, time_col], format='%H:%M:%S', errors='coerce')

                        if pd.notnull(t1) and pd.notnull(t2):
                            delta_sec = (t2 - t1).total_seconds()
                            delta_str = str(pd.to_timedelta(abs(delta_sec), unit='s')).split(' ')[-1]
                            formula = f"{t2.time()} - {t1.time()} = {delta_str}"
                            df.at[curr_idx, 'sec'] = abs(delta_sec)
                            df.at[curr_idx, 'min'] = delta_str
                            df.at[curr_idx, 'Reverse'] = formula
                            deltas.append(abs(delta_sec))

                    if deltas:
                        avg_sec = round(sum(deltas) / len(deltas), 4)
                        df.at[sub_group[-1], 'Average'] = f"Average-Group = {avg_sec} sec/strip"
                        df.at[sub_group[-1], 'Data Point'] = str(len(deltas))
                        all_deltas.extend(deltas)

            df['sec'] = pd.to_numeric(df['sec'], errors='coerce')
            df['SPEED'] = pd.to_numeric(df['SPEED'], errors='coerce')
            df['SPEED'] = df['SPEED'].replace(r'^\s*$', pd.NA, regex=True)
            df['Average_Frame-Stock'] = ''

            avg_all_sec = df['sec'].dropna().mean().round(4)
            df.at[0, 'Average_Frame-Stock'] = f"Average_All = {avg_all_sec} time/strip"

            def get_station_name(val):
                if pd.isna(val):
                    return None
                val_str = str(val)
                if '\\' in val_str:
                    val_str = val_str.split('\\')[-1]
                    val_str = re.sub(r'-[A-Z0-9]+$', '', val_str)
                return val_str.strip()
            df['__station__'] = df['Unnamed: 3'].apply(get_station_name)
            df['PACKAGE_CODE'] = df['__station__'].map(lambda x: item_map.get(str(x).strip()) if pd.notna(x) else None)

            # ==== เพิ่มส่วนตัด Outlier และใส่ข้อความ ====
            all_valid = df[
                df['__station__'].notna() &
                df['sec'].notna() &
                df['SPEED'].notna() &
                (df['SPEED'] != 0)
            ].copy()

            valid_groups = all_valid.copy()
            valid_groups['z_score'] = valid_groups.groupby(['__station__', 'SPEED'])['sec'].transform(lambda x: zscore(x, nan_policy='omit'))
            valid_groups = valid_groups[valid_groups['z_score'].abs() <= 3]

            outlier_info = {}
            grouped_all = all_valid.groupby(['__station__', 'SPEED'])
            grouped_valid = valid_groups.groupby(['__station__', 'SPEED'])

            for group_key, group_df in grouped_all:
                total = len(group_df)
                filtered = len(grouped_valid.get_group(group_key)) if group_key in grouped_valid.groups else 0
                msg = f"ไม่ตัด Outlier (ข้อมูลน้อย) — แถว: {total}" if total < 15 else f"ตัด Outlier — ก่อน: {total} หลัง: {filtered}"
                idx_first = group_df.index.min()
                outlier_info[idx_first] = msg

            df['outlier_removed'] = ""
            for idx, msg in outlier_info.items():
                df.at[idx, 'outlier_removed'] = msg

            # ==== ใช้ valid_groups ต่อ ====
            group_cols = ['__station__', 'SPEED']
            station_avg = valid_groups.groupby(group_cols)['sec'].mean().round(4).to_dict()

            first_indices = valid_groups.groupby(group_cols).apply(lambda g: g.index.min()).reset_index(name='idx')
            for row in first_indices.itertuples(index=False):
                station, speed, idx = row
                avg = station_avg.get((station, speed))
                if avg is not None:
                    df.at[idx, 'Average_Frame-Stock'] = f"{station}: Average = {avg} time/strip (SPEED={speed})"

            df.drop(columns=['__station__', 'DateOnly'], inplace=True)
            processed_df = df.copy()
            data = df.to_dict(orient='records')

    return render_template('index.html', files=files, data=data, selected_file=selected_file)


@app.route('/get_types', methods=['POST'])
def get_types():
    filename = request.json.get('filename')
    filepath = os.path.join(FRAMESTOCK_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify([])

    df = pd.read_excel(filepath)
    df = standardize_columns(df)
    op_col = next((c for c in df.columns if df[c].astype(str).str.contains('PRO|CUC', na=False).any()), None)
    if not op_col:
        return jsonify([])

    types = sorted(df[op_col].dropna().unique().tolist())
    return jsonify(types)

@app.route('/export_excel')
def export_excel():
    global processed_df
    selected_filename = request.args.get('filename')

    if processed_df is None or processed_df.empty:
        return "No data to export", 400

    if not selected_filename:
        return "No filename provided", 400

    col_op = 'Unnamed: 2'
    col_avg_info = 'Average'

    mask_pro = processed_df.get(col_op, pd.Series(dtype=str)).fillna('').astype(str).str.upper() == 'PRO'
    mask_avg = processed_df.get(col_avg_info, pd.Series(dtype=str)).fillna('').astype(str) != ''

    df_to_export = processed_df[mask_pro | mask_avg].copy()

    if 'min' in df_to_export.columns:
        bad_values = ['Restarted (Blank Row)', 'None', 'Machine Error', 'Restarted (Found 1 in Unnamed: 5)']
        mask_bad_min = (
            (df_to_export[col_op].fillna('').astype(str).str.upper() == 'PRO') &
            (df_to_export['min'].fillna('').astype(str).isin(bad_values))
        )
        df_to_export = df_to_export[~mask_bad_min]

    desired_columns = ['Unnamed: 3', 'SPEED', 'Average', 'Data Point', 'Average_Frame-Stock', 'PACKAGE_CODE', 'outlier_removed']
    df_to_export = df_to_export[[col for col in desired_columns if col in df_to_export.columns]]

    if 'Unnamed: 3' in df_to_export.columns:
         df_to_export['Unnamed: 3'] = df_to_export['Unnamed: 3'].astype(str).str.split('\\').str[-1]

    if df_to_export.empty:
        return "No matching data to export", 400

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_export.to_excel(writer, index=False, sheet_name='Processed Data')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=selected_filename
    )

@app.route('/export_all_pro')
def export_all_pro():
    global processed_df
    selected_filename = request.args.get('filename')

    if processed_df is None or processed_df.empty:
        return "No data to export", 400

    if not selected_filename:
        return "No filename provided", 400

    op_col = 'Unnamed: 2'
    info_col = 'Average_Info'

    mask_pro = processed_df.get(op_col, pd.Series(dtype=str)).fillna('').astype(str).str.upper() == 'PRO'
    mask_avg_all = processed_df.get(info_col, pd.Series(dtype=str)).fillna('').astype(str).str.contains('Average_All', na=False)

    df_to_export = processed_df[mask_pro | mask_avg_all].copy()

    columns_to_drop = [
        'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12',
        'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Speed'
    ]
    df_to_export.drop(columns=[col for col in columns_to_drop if col in df_to_export.columns], inplace=True)

    if 'Unnamed: 4' in df_to_export.columns:
        df_to_export['Unnamed: 4'] = df_to_export['Unnamed: 4'].astype(str).str.split('.').str[0]

    if df_to_export.empty:
        return "No PRO data found", 400

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_export.to_excel(writer, index=False, sheet_name='All PRO')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='all_pro_' + selected_filename
    )
#
if __name__ == '__main__':
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n✅ Flask app is running on: http://{ip}:8080\n(เปิดจากเครื่องอื่นในเครือข่ายได้ด้วย IP นี้)\n")
    app.run(debug=True, host='0.0.0.0', port=8080)

#test Gun