import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import re

class WireBondingAnalyzer:
    def __init__(self):
        self.nobump_df = None
        self.wb_data = None
        self.efficiency_df = None
        self.raw_data = None
    
    def normalize_model_name(self, model_name):
        """ทำความสะอาดและรวมชื่อรุ่นเครื่องที่คล้ายกัน"""
        if not isinstance(model_name, str):
            model_name = str(model_name)
        
        model_name = model_name.strip().upper()
        
        # รวม WB3100 ทุกเวอร์ชัน
        if 'WB3100' in model_name:
            return 'WB3100'
        
        # สามารถเพิ่มกฎการรวมรุ่นอื่นๆ ที่นี่
        if 'WB3200' in model_name:
            return 'WB3200'
        
        if 'WB3300' in model_name:
            return 'WB3300'
            
        return model_name

    def clean_model_names(self, df):
        """ทำความสะอาดชื่อรุ่นเครื่อง (เวอร์ชันปรับปรุง)"""
        df = df.copy()
        if 'machine model' in df.columns:
            df['machine model'] = df['machine model'].apply(self.normalize_model_name)
        return df
    
    def load_data(self, uph_path, wire_data_path):
        """โหลดข้อมูลที่จำเป็น"""
        try:
            # โหลดข้อมูล Wire Data
            self.nobump_df = pd.read_excel(wire_data_path)
            self.nobump_df.columns = self.nobump_df.columns.str.strip().str.upper()
            
            # โหลดข้อมูล UPH
            if uph_path.endswith('.csv'):
                self.raw_data = pd.read_csv(uph_path, encoding='utf-8-sig')
            else:
                self.raw_data = pd.read_excel(uph_path)
            
            # ทำความสะอาดคอลัมน์
            self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_wire_per_unit(self, bom_no):
        """คำนวณจำนวนสายต่อหน่วย"""
        try:
            bom_no = str(bom_no).strip().upper()
            bom_data = self.nobump_df[self.nobump_df['BOM_NO'].astype(str).str.strip().str.upper() == bom_no]
            
            if bom_data.empty:
                return 1.0
            
            no_bump = float(bom_data['NO_BUMP'].iloc[0]) if 'NO_BUMP' in bom_data.columns and not bom_data['NO_BUMP'].empty else 0
            num_required = float(bom_data['NUMBER_REQUIRED'].iloc[0]) if 'NUMBER_REQUIRED' in bom_data.columns and not bom_data['NUMBER_REQUIRED'].empty else 0
            
            wire_per_unit = (no_bump / 2) + num_required
            return wire_per_unit if wire_per_unit > 0 else 1.0
        except Exception as e:
            print(f"Error calculating wire per unit for BOM {bom_no}: {e}")
            return 1.0
    
    def remove_outliers(self, df):
        """ลบ outliers จากข้อมูลแบ่งตาม BOM และ Machine Model"""
        try:
            if df.empty:
                return df, {}
                
            df = self.clean_model_names(df)
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['uph', 'machine model', 'bom_no']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")
            
            # แบ่งข้อมูลตาม BOM และ Machine Model
            grouped = df.groupby(['bom_no', 'machine model'])
            cleaned_data = []
            outlier_info = {}
            
            for (bom_no, model), group_data in grouped:
                group_data = group_data.copy()
                original_count = len(group_data)
                
                # ข้ามถ้าข้อมูลน้อยกว่า 15 จุด
                if len(group_data) < 15:
                    cleaned_data.append(group_data)
                    outlier_info[(bom_no, model)] = {
                        'original_count': original_count,
                        'removed_count': 0,
                        'final_count': original_count
                    }
                    continue
                
                # กระบวนการตัด Outlier แบบอัตโนมัติ
                current_data = group_data
                
                for iteration in range(20):  # จำกัดจำนวนรอบ
                    # ใช้ Z-Score (±3σ)
                    z_threshold = 3
                    z_scores = zscore(current_data['uph'])
                    z_filtered = current_data[(z_scores >= -z_threshold) & (z_scores <= z_threshold)]
                    
                    # ตรวจสอบว่ายังมี Outlier หรือไม่
                    if not self._has_outliers(z_filtered['uph']):
                        current_data = z_filtered
                        break
                    
                    # ใช้ IQR (1.5*IQR)
                    Q1 = current_data['uph'].quantile(0.25)
                    Q3 = current_data['uph'].quantile(0.75)
                    IQR = Q3 - Q1
                    iqr_filtered = current_data[
                        (current_data['uph'] >= Q1 - 1.5*IQR) & 
                        (current_data['uph'] <= Q3 + 1.5*IQR)]
                    
                    if not self._has_outliers(iqr_filtered['uph']):
                        current_data = iqr_filtered
                        break
                    
                    current_data = iqr_filtered
                
                cleaned_data.append(current_data)
                final_count = len(current_data)
                
                # เก็บข้อมูลการตัด outlier
                outlier_info[(bom_no, model)] = {
                    'original_count': original_count,
                    'removed_count': original_count - final_count,
                    'final_count': final_count
                }
            
            result_df = pd.concat(cleaned_data) if cleaned_data else df
            return result_df, outlier_info
        
        except Exception as e:
            print(f"Error in remove_outliers: {e}")
            return df, {}
    
    def _has_outliers(self, series):
        """ตรวจสอบว่ายังมี Outlier หรือไม่"""
        if len(series) < 3:
            return False
        z_scores = zscore(series)
        return (abs(z_scores) > 3).any()
    
    def preprocess_data(self):
        """เตรียมข้อมูลก่อนคำนวณ"""
        try:
            if self.raw_data is None:
                raise ValueError("No data loaded")
            
            # คัดลอกข้อมูลและทำความสะอาด
            df = self.raw_data.copy()
            df.columns = df.columns.str.strip().str.lower()
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['uph', 'machine model', 'bom_no']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")
            
            # แปลงประเภทข้อมูล
            df['uph'] = pd.to_numeric(df['uph'], errors='coerce')
            df['bom_no'] = df['bom_no'].astype(str).str.strip().str.upper()
            
            # ลบแถวที่ไม่มีค่า UPH หรือ BOM_NO
            df = df.dropna(subset=['uph', 'bom_no'])
            
            # ทำความสะอาดชื่อรุ่นเครื่อง (เวอร์ชันปรับปรุง)
            df = self.clean_model_names(df)
            
            self.wb_data = df
            return True
        
        except Exception as e:
            print(f"Error in preprocess_data: {e}")
            return False
    
    def calculate_efficiency(self):
        """คำนวณประสิทธิภาพการทำงาน"""
        try:
            if not self.preprocess_data():
                return None
            
            # ตัด Outlier และเก็บข้อมูลการตัด
            cleaned_data, outlier_info = self.remove_outliers(self.wb_data)
            
            if cleaned_data.empty:
                return None
            
            # กลุ่มข้อมูลตาม BOM และรุ่นเครื่อง
            grouped = cleaned_data.groupby(['bom_no', 'machine model'])
            results = []
            
            for (bom_no, model), group in grouped:
                # คำนวณค่าเฉลี่ย UPH
                mean_uph = group['uph'].mean()
                # std_uph = group['uph'].std()  # ซ่อนการคำนวณ std_uph
                count = len(group)
                
                # คำนวณ Wire Per Unit
                wire_per_unit = self.calculate_wire_per_unit(bom_no)
                
                # คำนวณประสิทธิภาพ (UPH)
                efficiency = mean_uph / wire_per_unit if wire_per_unit > 0 else 0
                
                # ดึงข้อมูลเพิ่มเติม
                operation = group['operation'].iloc[0] if 'operation' in group.columns else 'N/A'
                optn_code = group['optn_code'].iloc[0] if 'optn_code' in group.columns else 'N/A'
                
                # ดึงข้อมูลการตัด outlier
                outlier_data = outlier_info.get((bom_no, model), {
                    'original_count': count,
                    'removed_count': 0,
                    'final_count': count
                })
                
                results.append({
                    'BOM': bom_no,
                    'Model': model,
                    'Operation': operation,
                    'Optn_Code': optn_code,
                    'Wire Per Hour': round(mean_uph, 2),
                    # 'Std_UPH': round(std_uph, 2),  # ซ่อนคอลัมน์ Std_UPH
                    'Wire_Per_Unit': round(wire_per_unit, 2),
                    'UPH': round(efficiency, 3),
                    'Data_Points': count,
                    'Original_Count': outlier_data['original_count'],
                    'Outliers_Removed': outlier_data['removed_count']
                })
            
            self.efficiency_df = pd.DataFrame(results)
            return self.efficiency_df
        
        except Exception as e:
            print(f"Error in calculate_efficiency: {e}")
            return None
    
    def generate_report(self):
        """สร้างรายงานสรุป"""
        if self.efficiency_df is None or self.efficiency_df.empty:
            return None
        
        try:
            # สรุปตามรุ่นเครื่อง
            model_summary = self.efficiency_df.groupby('Model').agg({
                'UPH': ['mean', 'std', 'count'],
                'Wire Per Hour': 'mean',
                'Wire_Per_Unit': 'mean'
            }).round(3)
            
            # สรุปทั้งหมด
            overall_stats = {
                'average_efficiency': round(self.efficiency_df['UPH'].mean(), 3),
                'average_wph': round(self.efficiency_df['Wire Per Hour'].mean(), 2),
                'best_model': self.efficiency_df.loc[self.efficiency_df['UPH'].idxmax()]['Model'],
                'best_efficiency': round(self.efficiency_df['UPH'].max(), 3),
                'worst_model': self.efficiency_df.loc[self.efficiency_df['UPH'].idxmin()]['Model'],
                'worst_efficiency': round(self.efficiency_df['UPH'].min(), 3),
                'total_boms': len(self.efficiency_df['BOM'].unique()),
                'total_models': len(self.efficiency_df['Model'].unique()),
                'total_data_points': self.efficiency_df['Data_Points'].sum()
            }
            
            report = {
                'overall': overall_stats,
                'by_model': model_summary.to_dict(),
                'details': self.efficiency_df.to_dict('records')
            }
            
            return report
        
        except Exception as e:
            print(f"Error in generate_report: {e}")
            return None
    
    def export_to_excel(self, file_path='wb_analysis_results.xlsx'):
        """ส่งออกผลลัพธ์เป็นไฟล์ Excel"""
        if self.efficiency_df is None or self.efficiency_df.empty:
            return False
        
        try:
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                # Sheet 1: ผลลัพธ์ UPH
                self.efficiency_df.to_excel(
                    writer, sheet_name='UPH_Results', index=False)
                
                # Sheet 2: สรุปตามรุ่นเครื่อง
                model_summary = self.efficiency_df.groupby('Model').agg({
                    'UPH': ['mean', 'std', 'count', 'min', 'max'],
                    'Wire Per Hour': 'mean',
                    'Wire_Per_Unit': 'mean'
                }).round(3)
                model_summary.to_excel(writer, sheet_name='Model_Summary')
                
                # Sheet 3: สรุปภาพรวม
                report = self.generate_report()
                if report and 'overall' in report:
                    overall_df = pd.DataFrame.from_dict(
                        report['overall'], orient='index', columns=['Value'])
                    overall_df.to_excel(writer, sheet_name='Overall_Summary')
            
            print(f"Exported results to {file_path}")
            return True
        
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    analyzer = WireBondingAnalyzer()
    
    # โหลดข้อมูล
    data_loaded = analyzer.load_data(
        uph_path='./data/Data WB UTL1 Jan-May-25.xlsx',  # เปลี่ยนเป็น path ของคุณ
        wire_data_path='./temp/Book6_Wire Data.xlsx'     # เปลี่ยนเป็น path ของคุณ
    )
    
    if data_loaded:
        # คำนวณประสิทธิภาพ
        efficiency_df = analyzer.calculate_efficiency()
        
        if efficiency_df is not None:
            # สร้างรายงาน
            report = analyzer.generate_report()
            
            # แสดงผลลัพธ์สรุป
            if report:
                print("\n=== สรุปผลการวิเคราะห์ ===")
                print(f"ประสิทธิภาพเฉลี่ย (UPH): {report['overall']['average_efficiency']}")
                print(f"รุ่นเครื่องที่ดีที่สุด: {report['overall']['best_model']} (UPH: {report['overall']['best_efficiency']})")
                print(f"จำนวนรุ่นเครื่องทั้งหมด: {report['overall']['total_models']}")
                print(f"จำนวน BOM ทั้งหมด: {report['overall']['total_boms']}")
            
            # ส่งออกไฟล์
            analyzer.export_to_excel('./results/wb_analysis.xlsx')
        else:
            print("ไม่สามารถคำนวณประสิทธิภาพได้")
    else:
        print("ไม่สามารถโหลดข้อมูลได้")